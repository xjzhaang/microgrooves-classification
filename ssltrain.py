import torch
import torchvision
import torch.nn as nn
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
from src_contrastive.model_contrastive import SimCLR, VICReg
from src_contrastive.trainer_contrastive import Trainer
from src_contrastive.losses import SupConLoss
from src_contrastive.transforms_contrastive import RandomGaussianNoise, RandomIntensityAdjust, RandomFrequencyMasking




def parse_args():
    parser = argparse.ArgumentParser(description='SimCLR Training with Fold Selection')

    # Training parameters
    parser.add_argument('--experiment', type=str, default='laminac_fibroblast_scaled',
                        help='Experiment name')
    parser.add_argument('--model', type=str, default='simclr', choices=['simclr', 'vicreg'],
                        help='model to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature parameter for NTXentLoss')
    parser.add_argument('--supervised', type=lambda x: x.lower() == 'true', default=False, 
                    help='Whether to use supervised contrastive loss')

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
        
def get_transform():
    # Expanded transform pipeline
    view_transform = torchvision.transforms.Compose([
        # Original transforms
        torchvision.transforms.Resize(size=512, interpolation=InterpolationMode.BILINEAR, antialias=True),
        # Modified with wider scale range as per SimCLR paper
        torchvision.transforms.RandomResizedCrop(size=512, scale=(0.8, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(p=0.7),
        torchvision.transforms.RandomVerticalFlip(p=0.7),
        torchvision.transforms.RandomRotation(180, interpolation=InterpolationMode.BILINEAR, expand=False,
                                              center=None, fill=0),

        # Added: Random affine for subtle distortions
        torchvision.transforms.RandomApply([
            torchvision.transforms.RandomAffine(
                degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5
            )
        ], p=0.3),

        # Modified: Apply Gaussian blur randomly instead of always
        # torchvision.transforms.RandomApply([
        #     torchvision.transforms.GaussianBlur(kernel_size=21, sigma=(0.1, 2.0))
        # ], p=0.5),

        # Convert to tensor before custom tensor transforms
        torchvision.transforms.ToTensor(),

        # Added: Custom tensor-based transforms
        RandomGaussianNoise(mean=0.0, std=0.05, p=0.3),
        RandomIntensityAdjust(factor_range=(0.7, 1.3), p=0.5),
        RandomFrequencyMasking(min_threshold=0.8, max_threshold=0.9999,p=0.3),
    ])

    return MultiViewTransform(transforms=[view_transform, view_transform])
        

def create_backbone(backbone_name):
    if backbone_name == 'resnet34':
        backbone = FFResNet34()
    else:  # default to resnet50
        backbone = FFResNet50()

    backbone.fc = torch.nn.Identity()
    #init_ffresnet_weights(backbone)
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
    log_dir = f"{args.log_dir}/{exp_name}_{fold_name}_{args.epochs}_{args.backbone + args.model}"
    save_path = f"{args.save_dir}/{args.experiment}/{fold_name}_{args.epochs}_{args.backbone + args.model}"

    # Ensure save directory existsdoes
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Compute dataset statistics

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


    backbone = create_backbone(args.backbone)

    if args.model == "simclr":
        if args.backbone == "resnet34":
            model = SimCLR(backbone, 512, 512, 128)
        elif args.backbone == "resnet50":
            model = SimCLR(backbone, 2048, 2048, 128)
    elif args.model == "vicreg":
        if args.backbone == "resnet34":
            model = VICReg(backbone)
        elif args.backbone == "resnet50":
            model = VICReg(backbone, input_dim = 2048,
                                    hidden_dim = 4096,
                                    output_dim = 4096,
                                    num_layers = 2,)

    if args.supervised:
        criterion = SupConLoss(temperature=args.temperature)
    else:
        if args.model == "simclr":
            criterion = lightlyloss.NTXentLoss(temperature=args.temperature)
            print("Using NTXentLoss!")
        elif args.model == "vicreg":
            criterion = lightlyloss.VICRegLoss()
            
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLRWarmup(optimizer, T_max=args.epochs, gamma=0.5, T_warmup=0)
    #scheduler = CosineAnnealingLRWarmup(optimizer, T_max=args.epochs,  T_warmup=5, min_lr=0.001)
    #scaler = torch.cuda.amp.GradScaler(init_scale=2 ** 16)
    scaler = torch.cuda.amp.GradScaler(
        init_scale=2**10,  # Start with a smaller scale factor (default is 2^16)
        growth_factor=1.5,  # Grow the scale more slowly (default is 2)
        backoff_factor=0.5,  # Reduce scale more gradually when NaNs occur
        growth_interval=2000  # Update less frequently (default is 2000
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