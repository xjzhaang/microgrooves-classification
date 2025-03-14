from pathlib import Path
from tqdm import tqdm
import torch
from monai.data import decollate_batch
from monai.transforms import Compose, Activations, AsDiscrete
from sklearn.metrics import classification_report
from monai.metrics import ROCAUCMetric
from torchvision.transforms import v2
from src.rmix import r_mix
from src.utils import set_deterministic_mode
from src.custom_transforms import BalancedSpectralMasking
import torch.nn.functional as F
import numpy as np
import skimage


class SaliencyPenaltyLoss(torch.nn.Module):
    def __init__(self):
        super(SaliencyPenaltyLoss, self).__init__()
    
    def forward(self, saliency, mask):
        # Create the penalization mask
        saliency_min = saliency.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        saliency_max = saliency.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        normalized_saliency = (saliency - saliency_min) / (saliency_max - saliency_min + 1e-8)  

        penalization_mask = 1 - mask  # Inverts the mask (1 -> 0, 0 -> 1)

        # Compute the penalization loss
        penalization_loss = F.mse_loss(normalized_saliency * penalization_mask, torch.zeros_like(normalized_saliency), reduction='sum') 

        #Normalize the loss by the number of penalized pixels to avoid scale issues
        num_penalized_pixels = penalization_mask.sum()
        if num_penalized_pixels > 0:
            penalization_loss /= num_penalized_pixels

        return penalization_loss
        
class DistancePenalizedSaliencyLoss(torch.nn.Module):
    def __init__(self):
        super(DistancePenalizedSaliencyLoss, self).__init__()

    def forward(self, saliency, mask):
        # Convert mask to numpy array for distance transform
        saliency_min = saliency.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        saliency_max = saliency.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        normalized_saliency = (saliency - saliency_min) / (saliency_max - saliency_min + 1e-8)  
        
        penalized_saliency = normalized_saliency * mask
        # skimage.io.imsave("normalized_saliency.tif", normalized_saliency.to("cpu").numpy())
        # skimage.io.imsave("penalized_saliency.tif", penalized_saliency.to("cpu").numpy())

        # Compute the penalized loss (e.g., using mean squared error)
        penalized_loss = F.mse_loss(penalized_saliency, torch.zeros_like(penalized_saliency)) 

        return penalized_loss
        

class Wrapper:
    def __init__(self, model, num_classes):
        self.model = model
        self.num_classes = num_classes

    def __call__(self, x):
        return self.model(x).view(-1, self.num_classes, *([1] * 2))
        

class Trainer():
    def __init__(self, train_loader, val_loader, model, optimizer, scheduler, scaler, loss_fn, loss_fn_val, num_classes,
                 device, save_path, writer, epochs):
        self.dataloader_train = train_loader
        self.dataloader_val = val_loader
        self.model = model.to(device)
        self.device = device
        self.loss_function = loss_fn.to(self.device)
        self.saliency_loss = DistancePenalizedSaliencyLoss() #SaliencyPenaltyLoss()
        self.epochs = epochs
        self.writer = writer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs_run = 0
        self.train_loss_ls = []
        self.test_loss_ls = []
        self.best_loss = 999
        self.iqr_loss = 9999
        self.std_loss = 9999
        self.best_f1 = -1
        self.scaler = scaler
        self.save_path = save_path
        self.num_classes = num_classes
        self.loss_fn_val = loss_fn_val.to(self.device)
        self.auc_metric = ROCAUCMetric(average="weighted")

        if (Path(self.save_path) / "model.pth").exists():
            # Load the checkpoint files
            checkpoint = torch.load(Path(self.save_path) / "model.pth", map_location=device)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.epochs_run = checkpoint['epochs_run']
            self.scaler.load_state_dict(checkpoint['scaler'])
            self.best_loss = checkpoint['loss']
            self.best_f1 = checkpoint['f1']

    def train(self):
        Path(self.save_path).mkdir(parents=True, exist_ok=True)

        for epoch in range(self.epochs_run, self.epochs):
            train_loss, lr = self.train_step()
            if epoch % 2 == 1:
                loss, iqr_loss, std_loss, f1 = self.validation_step()

            self.epochs_run += 1

            if epoch % 2 == 1:
                state = dict(
                    epoch=epoch + 1,
                    model=self.model.state_dict(),
                    optimizer=self.optimizer.state_dict(),
                    epochs_run=self.epochs_run,
                    scheduler=self.scheduler.state_dict(),
                    scaler=self.scaler.state_dict(),
                    loss=min(self.best_loss, loss),
                    iqr_loss=self.iqr_loss,
                    std_loss=self.std_loss,
                    f1=self.best_f1,
                )
                if self.best_loss >= loss and self.iqr_loss > iqr_loss:# and self.std_loss > std_loss:
                    self.best_loss = loss
                    self.iqr_loss = iqr_loss
                    self.std_loss = std_loss
                    torch.save(state, Path(self.save_path) / f"best_loss.pth")
                if self.best_loss >= loss:
                    self.best_loss = loss
                    torch.save(state, Path(self.save_path) / f"best_loss2.pth")
                elif self.best_f1 <= f1:
                    self.best_f1 = f1
                    torch.save(state, Path(self.save_path) / f"best_f1.pth")

            state = dict(
                epoch=epoch + 1,
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                epochs_run=self.epochs_run,
                scheduler=self.scheduler.state_dict(),
                scaler=self.scaler.state_dict(),
                loss=self.best_loss,
                iqr_loss=self.iqr_loss,
                std_loss=self.std_loss,
                f1=self.best_f1,
            )
            torch.save(state, Path(self.save_path) / "model.pth")


    def train_step(self):
        loss = 0
        self.model.train()
        progress_bar = tqdm(enumerate(self.dataloader_train), total=len(self.dataloader_train),
                            desc=f"Epoch #{self.epochs_run}")
        
        #cutmix = v2.CutMix(alpha=0.5, num_classes=self.num_classes)
        saliency_loss_track = 0
        #spectral_transform = BalancedSpectralMasking(num_band=24)
        for batch, data in progress_bar:
            X, y, mask = data["image"].to(self.device), data["label"].to(self.device), data["mask"].to(self.device)
            ##Cutmix
            #X_masked = spectral_transform([X])[0]
            #X, y = cutmix(X, y)
            # skimage.io.imsave("X.tif", X.to("cpu").numpy())
            # skimage.io.imsave("mask.tif", mask.to("cpu").numpy().astype("uint16"))
            
            ##R-mix
            #y_onehot = F.one_hot(y, self.num_classes)
            #saliency = r_mix(model=self.model, inputs=X, labels=y_onehot, num_patches=32)
            #print(saliency.shape, saliency.max(), saliency.min())
            with torch.cuda.amp.autocast():
                y_pred = self.model(X)
                loss_batch = self.loss_function(y_pred, y)# + 0.01 * self.saliency_loss(saliency, mask)
                #saliency_loss_track+=0.01 * self.saliency_loss(saliency, mask)

            grad_stats = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"NaN gradient found in {name} at batch {batch}")
                        # Print parameter statistics to help identify issues
                        print(f"  Parameter stats - min: {param.min().item()}, max: {param.max().item()}, "
                              f"mean: {param.mean().item()}, std: {param.std().item()}")
                    else:
                        # Collect gradient statistics
                        grad_norm = torch.norm(param.grad).item()
                        grad_stats[name] = grad_norm
            
            if batch % 30 == 0:  # Only print occasionally to avoid clutter
                largest_grads = sorted(grad_stats.items(), key=lambda x: x[1], reverse=True)[:5]
                print("Largest gradient norms:")
                for name, norm in largest_grads:
                    print(f"  {name}: {norm}")
                    
            self.optimizer.zero_grad()
            self.scaler.scale(loss_batch).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss += loss_batch.item()
            lr = self.scheduler.get_last_lr()[0]

            progress_bar.set_postfix_str(
                f"Current loss {loss_batch.item():.5f} | Train loss {loss / (batch + 1):.5f} | Learning rate {lr:.6f}")
        #print(saliency_loss_track / (batch + 1))
        self.scheduler.step()
        loss = loss / len(self.dataloader_train)
        self.writer.add_scalar('Trackers/train loss', loss, self.epochs_run)
        self.writer.add_scalar('Trackers/learning rate', lr, self.epochs_run)

        return loss, lr

    def validation_step(self):
        loss = 0
        loss_l = []
        self.model.eval()
        y_pred_trans = Compose([Activations(softmax=True)])
        y_trans = Compose([AsDiscrete(to_onehot=self.num_classes)])

        class_loss = [0.0] * self.num_classes
        class_counts = [0] * self.num_classes
        #set_deterministic_mode(seed=0)
        with torch.no_grad():
            y_pred_agg = torch.tensor([], dtype=torch.float32, device=self.device)
            y_agg = torch.tensor([], dtype=torch.long, device=self.device)
            progress_bar = tqdm(enumerate(self.dataloader_val), total=len(self.dataloader_val),
                                desc=f"Epoch #{self.epochs_run}")
            for batch, data in progress_bar:
                X, y = data["image"].to(self.device).to(torch.float32), data["label"].to(self.device)
                #skimage.io.imsave("X.tif", X.to("cpu").numpy())
                with torch.cuda.amp.autocast():
                    y_pred = self.model(X)
                    loss_batch = self.loss_fn_val(y_pred, y)

                    for i in range(self.num_classes):
                        mask = (y == i).float()
                        class_loss[i] += ((mask * loss_batch)).sum().item()
                        class_counts[i] += mask.sum().item()
                        
                loss += loss_batch.item()
                loss_l.append(loss_batch.item())

                y_pred_agg = torch.cat([y_pred_agg, y_pred], dim=0)
                y_agg = torch.cat([y_agg, y], dim=0)

                lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix_str(
                    f"Current loss {loss_batch.item():.5f} |   Val loss {loss / (batch + 1):.5f} | Learning rate {lr:.6f}")

            y_onehot = [y_trans(i).cpu() for i in decollate_batch(y_agg, detach=False)]
            y_pred_act = [y_pred_trans(i).cpu() for i in decollate_batch(y_pred_agg)]
            self.auc_metric(y_pred_act, y_onehot)
            auc_res = self.auc_metric.aggregate()
            self.auc_metric.reset()
            del y_pred_act, y_onehot
            class_rep = classification_report(y_agg.cpu().numpy(), y_pred_agg.argmax(dim=1).cpu().numpy(), digits=4, output_dict=True, zero_division=0)
            macro_f1 = class_rep["macro avg"]["f1-score"]

            acc_metric = class_rep["accuracy"]
            loss = loss / len(self.dataloader_val)
            validation_losses = np.array(loss_l)
            std_loss = np.std(validation_losses)

            class_means = [class_loss[i] / class_counts[i] if class_counts[i] != 0 else 0 for i in range(self.num_classes)]  # Compute mean class losses
            class_loss_ = np.max(class_means) + np.min(class_means) / 2
            
            print(loss, class_loss_, std_loss, class_means)

            self.writer.add_scalar('Score/val loss', loss, self.epochs_run)
            self.writer.add_scalar('Score/auc', auc_res, self.epochs_run)
            self.writer.add_scalar('Score/accuracy', acc_metric, self.epochs_run)
            self.writer.add_scalar('Score/macro_f1', macro_f1, self.epochs_run)

        return loss, class_loss_, std_loss, macro_f1

    