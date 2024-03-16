from pathlib import Path
from tqdm import tqdm
import torch
from monai.data import decollate_batch
from monai.transforms import Compose, Activations, AsDiscrete
from sklearn.metrics import classification_report
from monai.metrics import ROCAUCMetric


def sliding_window_inferencer(image, model, patch_size=(1000, 1000), overlap=0.7, num_classes=5):
    """
    Perform sliding window inference on a large image using PyTorch.

    Parameters:
        image (numpy.ndarray or torch.Tensor): The large input image.
        model: The classification model.
        patch_size (tuple): Size of the patches for inference.
        overlap (float): Overlap fraction between patches.

    Returns:
        numpy.ndarray: The average classification tensor.
    """

    image_height, image_width = image.shape[-2:]

    # Convert patch_size to torch.Tensor if it's a tuple
    if isinstance(patch_size, tuple):
        patch_size = torch.tensor(patch_size, device="cuda")

    # Calculate overlap pixels
    overlap_pixels_y = int(patch_size[0] * overlap)
    overlap_pixels_x = int(patch_size[1] * overlap)

    # Initialize empty classification tensor
    #classification_tensor = torch.zeros(1, num_classes, dtype=torch.float32, device="cuda")
    classification_tensor = torch.full((1, num_classes), -999, dtype=torch.float32, device="cuda")
    # Initialize a counter tensor to keep track of the number of predictions per pixel
    counter_tensor = 0

    # Slide the window and perform inference
    for y in range(0, image_height - patch_size[0] + 1, overlap_pixels_y):
        for x in range(0, image_width - patch_size[1] + 1, overlap_pixels_x):
            # Extract patch
            patch = image[..., y:y+patch_size[0], x:x+patch_size[1]]

            # Perform inference on the patch
            with torch.cuda.amp.autocast():
                classification = model(patch)
            # Update the classification tensor
            classification_tensor = torch.max(classification_tensor, classification)

            # Update the counter tensor
    #         counter_tensor += 1

    # # Divide the classification tensor by the number of overlapping patches
    # classification_tensor /= counter_tensor

    return classification_tensor


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
        self.epochs = epochs
        self.writer = writer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs_run = 0
        self.train_loss_ls = []
        self.test_loss_ls = []
        self.best_loss = 999
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
                test_loss, f1_weighted = self.validation_step()

            self.epochs_run += 1

            state = dict(
                epoch=epoch + 1,
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                epochs_run=self.epochs_run,
                scheduler=self.scheduler.state_dict(),
                scaler=self.scaler.state_dict(),
                loss=self.best_loss,
                f1=self.best_f1,
            )
            torch.save(state, Path(self.save_path) / "model.pth")

            if epoch % 2 == 1:
                state = dict(
                    epoch=epoch + 1,
                    model=self.model.state_dict(),
                    optimizer=self.optimizer.state_dict(),
                    epochs_run=self.epochs_run,
                    scheduler=self.scheduler.state_dict(),
                    scaler=self.scaler.state_dict(),
                    loss=self.best_loss,
                    f1=self.best_f1,
                )
                if self.best_loss >= test_loss:
                    self.best_loss = test_loss
                    torch.save(state, Path(self.save_path) / f"best_loss.pth")
                # elif self.best_f1 <= f1_weighted:
                #     self.best_f1 = f1_weighted
                #     torch.save(state, Path(self.save_path) / f"best_f1.pth")

    def train_step(self):
        loss = 0
        self.model.train()
        progress_bar = tqdm(enumerate(self.dataloader_train), total=len(self.dataloader_train),
                            desc=f"Epoch #{self.epochs_run}")
        for batch, data in progress_bar:
            X, y = data["image"].to(self.device), data["label"].to(self.device)
            with torch.cuda.amp.autocast():
                y_pred = self.model(X)

                loss_batch = self.loss_function(y_pred, y)

            self.optimizer.zero_grad()
            self.scaler.scale(loss_batch).backward()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss += loss_batch.item()
            lr = self.scheduler.get_last_lr()[0]

            progress_bar.set_postfix_str(
                f"Current loss {loss_batch.item():.5f} | Train loss {loss / (batch + 1):.5f} | Learning rate {lr:.6f}")
        self.scheduler.step()
        loss = loss / len(self.dataloader_train)
        self.writer.add_scalar('Trackers/train loss', loss, self.epochs_run)
        self.writer.add_scalar('Trackers/learning rate', lr, self.epochs_run)

        return loss, lr

    def validation_step(self):
        loss = 0
        self.model.eval()
        y_pred_trans = Compose([Activations(softmax=True)])
        y_trans = Compose([AsDiscrete(to_onehot=self.num_classes)])
        with torch.no_grad():
            y_pred_agg = torch.tensor([], dtype=torch.float32, device=self.device)
            y_agg = torch.tensor([], dtype=torch.long, device=self.device)
            progress_bar = tqdm(enumerate(self.dataloader_val), total=len(self.dataloader_val),
                                desc=f"Epoch #{self.epochs_run}")
            for batch, data in progress_bar:
                X, y = data["image"].to(self.device).to(torch.float32), data["label"].to(self.device)
                with torch.cuda.amp.autocast():
                    y_pred = self.model(X)
                    loss_batch = self.loss_fn_val(y_pred, y)  # Convert y to float for BCELoss
                loss += loss_batch.item()

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

            self.writer.add_scalar('Score/val loss', loss, self.epochs_run)
            self.writer.add_scalar('Score/auc', auc_res, self.epochs_run)
            self.writer.add_scalar('Score/accuracy', acc_metric, self.epochs_run)
            self.writer.add_scalar('Score/macro_f1', macro_f1, self.epochs_run)

        return loss, macro_f1

    