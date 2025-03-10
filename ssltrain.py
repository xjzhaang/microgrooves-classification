import torch
import torchvision
from lightly import loss as lightlyloss

from torch.utils.data import DataLoader
from lightly.transforms.multi_view_transform import MultiViewTransform

from src.dataset import MyoblastDataset
from src.utils import compute_mean_std
from src.optimizers import StepLRWarmup
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import InterpolationMode
import random

from src.resnets import FFResNet50, FFResNet34
from src_contrastive.dataset_contrastive import SimCLRDataset
from src_contrastive.model_contrastive import SimCLR
from src_contrastive.trainer_contrastive import Trainer



EXPERIMENT="laminac_fibroblast_scaled"
EPOCH=200
EXP_IDS = [240215, 240219, 240805, 240807,]
EXP_IDS_VAL = [240221, 240731, 240802]
# EXP_IDS = [240221, 240219, 240731, 240802,]
# EXP_IDS_VAL = [240215, 240805, 240807]


LOG_DIR = f"{EXPERIMENT}_{EXP_IDS}_{EPOCH}"
SAVE_PATH = f"pretrain/{EXPERIMENT}/{EXP_IDS}_{EPOCH}"

ds_dataset = MyoblastDataset(cell_type=EXPERIMENT, exp_ids=EXP_IDS, mode="cropped", transform=None)
mean, std = compute_mean_std(ds_dataset)
print(mean, std)


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

    # Added: Color/intensity transforms
    # torchvision.transforms.RandomApply([
    #     torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5)
    # ], p=0.8),

    # Convert to tensor before custom tensor transforms
    torchvision.transforms.ToTensor(),

    # Added: Custom tensor-based transforms
    RandomSolarize(threshold=0.5, p=0.3),
    RandomGaussianNoise(mean=0.0, std=0.05, p=0.3),
    RandomIntensityAdjust(factor_range=(0.7, 1.3), p=0.5),
])

transform = MultiViewTransform(transforms=[view_transform, view_transform])
dataset_train = SimCLRDataset(cell_type=EXPERIMENT, exp_ids=EXP_IDS, mode="train", transform=transform)
dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True, drop_last=True, num_workers=4)


#backbone = torchvision.models.resnet34()
backbone = FFResNet50()
backbone.fc = torch.nn.Identity()

# Build the SimCLR model.
model = SimCLR(backbone)
criterion = lightlyloss.NTXentLoss(temperature=0.1)
optimizer = torch.optim.AdamW(model.parameters(), 0.001)#, eps=1e-05)
scheduler = StepLRWarmup(optimizer, T_max=EPOCH,  gamma=0.5, T_warmup=0)
#scheduler = CosineAnnealingLRWarmup(optimizer, T_max=EPOCH,  T_warmup=5)
scaler = torch.cuda.amp.GradScaler(init_scale=2**14,)

writer = SummaryWriter(log_dir=f"pretrain/runs/{LOG_DIR}")
trainer = Trainer(dataloader_train, model, optimizer, scheduler, scaler, criterion, device='cuda', save_path=SAVE_PATH, writer=writer, epochs=EPOCH)
trainer.train()
writer.close()
