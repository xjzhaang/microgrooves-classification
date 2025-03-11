import torch
from pathlib import Path
from tqdm import tqdm

class Trainer():
    def __init__(self, train_loader, model, optimizer, scheduler, scaler, loss_fn,
                 device, save_path, writer, epochs, supervised=False):
        self.dataloader_train = train_loader
        self.model = model.to(device)
        self.device = device
        self.loss_function = loss_fn.to(self.device)
        self.epochs = epochs
        self.writer = writer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs_run = 0
        self.train_loss_ls = []
        self.scaler = scaler
        self.save_path = save_path
        self.best_loss = 999
        self.supervised = supervised

        if (Path(self.save_path) / "model.pth").exists():
            # Load the checkpoint files
            checkpoint = torch.load(Path(self.save_path) / "model.pth", map_location=device)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.epochs_run = checkpoint['epochs_run']
            self.scaler.load_state_dict(checkpoint['scaler'])

    def train(self):
        Path(self.save_path).mkdir(parents=True, exist_ok=True)

        for epoch in range(self.epochs_run, self.epochs):
            loss, lr = self.train_step()
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
                )
                if self.best_loss >= loss:
                    self.best_loss = loss
                    torch.save(state, Path(self.save_path) / f"best_loss.pth")

            state = dict(
                epoch=epoch + 1,
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                epochs_run=self.epochs_run,
                scheduler=self.scheduler.state_dict(),
                scaler=self.scaler.state_dict(),
                loss=self.best_loss,
            )
            torch.save(state, Path(self.save_path) / "model.pth")

    def train_step(self, accumulation_steps=16):
        loss = 0
        self.model.train()
        self.optimizer.zero_grad()
        progress_bar = tqdm(enumerate(self.dataloader_train), total=len(self.dataloader_train),
                            desc=f"Epoch #{self.epochs_run}", ascii=True)
        for batch, data in progress_bar:
            view0, view1, labels = data["image"][0].to(self.device), data["image"][1].to(self.device), data["label"].to(self.device)
            with torch.cuda.amp.autocast():
                z0 = self.model(view0)
                z1 = self.model(view1)
                
                if self.supervised:
                    # Stack embeddings from both views
                    # Reshape to [batch_size, n_views, feature_dim]
                    features = torch.cat([z0.unsqueeze(1), z1.unsqueeze(1)], dim=1)
                    loss_batch = self.loss_function(features, labels) / accumulation_steps
                else:
                    # Original SimCLR loss
                    loss_batch = self.loss_function(z0, z1) / accumulation_steps

            self.scaler.scale(loss_batch).backward()
            loss += loss_batch.item() * accumulation_steps  # Adjust for reporting actual loss
            if (batch + 1) % accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            lr = self.scheduler.get_last_lr()[0]

            progress_bar.set_postfix_str(
                f"Current loss {loss_batch.item():.5f} | Train loss {loss / (batch + 1):.5f} | Learning rate {lr:.6f}")
        self.scheduler.step()
        loss = loss / len(self.dataloader_train)
        self.writer.add_scalar('Trackers/train loss', loss, self.epochs_run)
        self.writer.add_scalar('Trackers/learning rate', lr, self.epochs_run)

        return loss, lr