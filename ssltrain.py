import torch
import torchvision
from lightly import loss as lightlyloss
import argparse
import os

from torch.utils.data import DataLoader
from lightly.transforms.multi_view_transform import MultiViewTransform

from src.dataset import MyoblastDataset
from src.utils import compute_mean_std
from src.optimizers import StepLRWarmup, CosineAnnealingLRWarmup
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import InterpolationMode
import random

from src.resnets import FFResNet50, FFResNet34
from src_contrastive.dataset_contrastive import SimCLRDataset
from src_contrastive.model_contrastive import SimCLR
from src_contrastive.trainer_contrastive import Trainer
from src_contrastive.losses import SupConLoss




def parse_args():
    parser = argparse.ArgumentParser(description='SimCLR Training with Fold Selection')

    # Training parameters
    parser.add_argument('--experiment', type=str, default='laminac_fibroblast_scaled',
                        help='Experiment name')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature parameter for NTXentLoss')
    parser.add_argument('--supervised', type=bool, default=False,
                        help='Supervised contrastive learning')

    # Fold selection
    parser.add_argument('--fold', type=int, default=1, choices=[1, 2],
                        help='Fold number for train/val split (1, 2, or 3)')

    # Model selection
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet34', 'resnet50'],
                        help='Backbone architecture')

    # Other parameters
    parser.add_argument('--log_dir', type=str, default='pretrain/runs',
                        help='Directory for tensorboard logs')
    parser.add_argument('--save_dir', type=str, default='pretrain',
                        help='Directory to save models')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of data loading workers')

    return parser.parse_args()


# Predefined fold configurations
FOLD_CONFIGS = {
    1: {
        'train': [240215, 240219, 240805, 240807],
        'val': [240221, 240731, 240802]
    },
    2: {
        'train': [240221, 240219, 240731, 240802],
        'val': [240215, 240805, 240807]
    },
}


class RandomSolarize(torch.nn.Module):
    def __init__(self, threshold=0.5, p=0.3):
        super().__init__()
        self.threshold = threshold
        self.p = p

    def forward(self, img):
        if random.random() < self.p:
            return torch.where(img >= self.threshold, 1.0 - img, img)
        return img


# Custom Gaussian noise transform
class RandomGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0.0, std=0.1, p=0.3):
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p

    def forward(self, img):
        if random.random() < self.p:
            noise = torch.randn_like(img) * self.std + self.mean
            return torch.clamp(img + noise, 0, 1)
        return img


# Custom intensity adjustment transform
class RandomIntensityAdjust(torch.nn.Module):
    def __init__(self, factor_range=(0.7, 1.3), p=0.5):
        super().__init__()
        self.factor_range = factor_range
        self.p = p

    def forward(self, img):
        if random.random() < self.p:
            factor = random.uniform(*self.factor_range)
            return torch.clamp(img * factor, 0, 1)
        return img


def get_transform():
    # Expanded transform pipeline
    view_transform = torchvision.transforms.Compose([
        # Original transforms
        torchvision.transforms.Resize(size=512, interpolation=InterpolationMode.BILINEAR, antialias=True),
        # Modified with wider scale range as per SimCLR paper
        torchvision.transforms.RandomResizedCrop(size=512, scale=(0.8, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(p=0.7),
        torchvision.transforms.RandomVerticalFlip(p=0.7),
        torchvision.transforms.RandomRotation((-180, 180), interpolation=InterpolationMode.BILINEAR, expand=False,
                                              center=None, fill=0),

        # Added: Random affine for subtle distortions
        torchvision.transforms.RandomApply([
            torchvision.transforms.RandomAffine(
                degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5
            )
        ], p=0.3),

        # Modified: Apply Gaussian blur randomly instead of always
        torchvision.transforms.RandomApply([
            torchvision.transforms.GaussianBlur(kernel_size=21, sigma=(0.1, 2.0))
        ], p=0.5),

        # Convert to tensor before custom tensor transforms
        torchvision.transforms.ToTensor(),

        # Added: Custom tensor-based transforms
        RandomSolarize(threshold=0.5, p=0.3),
        RandomGaussianNoise(mean=0.0, std=0.05, p=0.3),
        RandomIntensityAdjust(factor_range=(0.7, 1.3), p=0.5),
    ])

    return MultiViewTransform(transforms=[view_transform, view_transform])


def create_backbone(backbone_name):
    if backbone_name == 'resnet34':
        backbone = FFResNet34()
    else:  # default to resnet50
        backbone = FFResNet50()

    backbone.fc = torch.nn.Identity()
    return backbone


def main():
    args = parse_args()

    # Get fold configuration
    fold_config = FOLD_CONFIGS[args.fold]
    train_exp_ids = fold_config['train']
    val_exp_ids = fold_config['val']

    print(f"Using fold {FOLD_CONFIGS[args.fold]}:")
    print(f"  Train experiments: {train_exp_ids}")
    print(f"  Validation experiments: {val_exp_ids}")

    # Create directory paths and names
    exp_name = f"{args.experiment}"
    fold_name = FOLD_CONFIGS[args.fold]["train"]
    log_dir = f"{args.log_dir}/{exp_name}_{fold_name}_{args.epochs}"
    save_path = f"{args.save_dir}/{args.experiment}/{fold_name}_{args.epochs}"

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Compute dataset statistics
    ds_dataset = MyoblastDataset(cell_type=args.experiment, exp_ids=train_exp_ids, mode="cropped", transform=None)
    mean, std = compute_mean_std(ds_dataset)
    print(f"Dataset mean: {mean}, std: {std}")

    # Create transforms and dataset
    transform = get_transform()
    dataset_train = SimCLRDataset(
        cell_type=args.experiment,
        exp_ids=train_exp_ids,
        mode="train",
        transform=transform
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers
    )

    # Create model and training components
    backbone = create_backbone(args.backbone)
    model = SimCLR(backbone, 512, 512, 128)
    
    if args.supervised:
        criterion = SupConLoss(temperature=args.temperature)
    else:
        criterion = lightlyloss.NTXentLoss(temperature=args.temperature)
    optimizer = torch.optim.AdamW(model.parameters(), args.lr)
    #scheduler = StepLRWarmup(optimizer, T_max=args.epochs, gamma=0.5, T_warmup=0)
    scheduler = CosineAnnealingLRWarmup(optimizer, T_max=args.epochs,  T_warmup=5)
    #scaler = torch.cuda.amp.GradScaler(init_scale=2 ** 14)
    scaler = torch.cuda.amp.GradScaler(
        init_scale=2**10,  # Start with a smaller scale factor (default is 2^16)
        growth_factor=1.5,  # Grow the scale more slowly (default is 2)
        backoff_factor=0.5,  # Reduce scale more gradually when NaNs occur
        growth_interval=100  # Update less frequently (default is 2000
        )
    # Set up tensorboard and trainer
    writer = SummaryWriter(log_dir=log_dir)

    trainer = Trainer(
        dataloader_train,
        model,
        optimizer,
        scheduler,
        scaler,
        criterion,
        device='cuda',
        save_path=save_path,
        writer=writer,
        epochs=args.epochs,
        supervised=args.supervised,
    )

    # Train the model
    trainer.train()
    writer.close()


if __name__ == "__main__":
    main()