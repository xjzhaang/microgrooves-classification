import torch
import numpy as np
from monai.transforms import Compose, RandRotated, RandRotate90d, Resized, RandZoomd, RandAdjustContrastd, RandCropByPosNegLabeld, NormalizeIntensityd, RandFlipd, RandShiftIntensityd, RandGaussianNoised, RandHistogramShiftd, ScaleIntensityRanged
from monai.transforms import CutMixd
from monai.utils import set_determinism
from src.custom_transforms import ThresholdMaskingd, CropAndStackTransformd, MIPTransformd

def compute_mean_std(dataset):
    """
    Compute the mean and standard deviation for a given dataset.

    Args:
    - dataset: PyTorch dataset containing images

    Returns:
    - mean: Mean values per channel of the dataset
    - std: Standard deviations per channel of the dataset
    """
    # Initialize lists to store per-channel mean and std
    mean_per_channel = []
    std_per_channel = []

    # Iterate over the dataset
    for i in range(len(dataset)):
        image = dataset[i]["image"]

        # Compute mean and std per channel
        mean_channel = torch.mean(image, dim=(1, 2))
        std_channel = torch.std(image, dim=(1, 2))

        # Append mean and std values per channel to lists
        mean_per_channel.append(mean_channel)
        std_per_channel.append(std_channel)

    # Stack the per-channel mean and std tensors along the 0th dimension
    mean = torch.stack(mean_per_channel, dim=0)
    std = torch.stack(std_per_channel, dim=0)

    # Compute the overall mean and std across all samples
    mean = torch.mean(mean, dim=0)
    std = torch.mean(std, dim=0)

    # Convert to numpy arrays for easier handling
    mean = mean.numpy()
    std = std.numpy()

    return np.array(mean), np.array(std)
    

def create_transforms(mean, std):
    train_transforms = Compose(
        [
            Resized(
                keys=["image", "mask"],
                spatial_size=(512, 512),
                mode="bilinear",
            ),
            
            # MIPTransformd(
            #     keys=["image"],
            # ),
            RandFlipd(
                keys=["image", "mask"],
                prob=0.7,
                spatial_axis=1,
            ),
            RandFlipd(
                keys=["image", "mask"],
                prob=0.7,
                spatial_axis=0,
            ),
            RandRotate90d(
                keys=["image", "mask"],
                prob=0.7,
                max_k=3,
            ),
            RandRotated(
                keys=["image", "mask"],
                range_x=3.14,
                prob=1,
                padding_mode="zeros",
            ),
            # RandZoomd(
            #     keys=["image", "mask"],
            #     min_zoom=0.9,
            #     max_zoom=1.1,
            #     prob=0.3,
            # ),
            # RandShiftIntensityd(
            #     keys=["image"],
            #     prob=0.3,
            #     offsets=0.1,
            # ),
            RandAdjustContrastd(
                keys=["image"],
                prob=0.4,
                gamma=(1.7, 2.4),
            ),
            # NormalizeIntensityd(
            #     keys=["image"],
            #     # subtrahend=mean,
            #     # divisor=std,
            #     #nonzero=True,
            #     channel_wise=True,
            # ),
            # ThresholdMaskingd(
            #     keys=["image"],
            # ),
        ]
    )
    val_transforms = Compose(
        [
            Resized(
                keys=["image"],
                spatial_size=(512, 512),
                mode="bilinear",
            ),
            # MIPTransformd(
            #     keys=["image"],
            # ),
            # RandShiftIntensityd(
            #     keys=["image"],
            #     prob=0.5,
            #     offsets=0.1,
            # ),
            # RandAdjustContrastd(
            #     keys=["image"],
            #     prob=1,
            #     gamma=(1.7, 2.4),
            # ),
            # NormalizeIntensityd(
            #     keys=["image"],
            #     # subtrahend=mean,
            #     # divisor=std,
            #     #nonzero=True,
            #     channel_wise=True,
            # ),
            # ThresholdMaskingd(
            #     keys=["image"],
            # ),
        ]
    )
    test_transforms = Compose(
        [
            Resized(
                keys=["image"],
                spatial_size=(1024, 1024),
                mode="bilinear",
            ),
            # NormalizeIntensityd(
            #     keys=["image"],
            #     # subtrahend=mean,
            #     # divisor=std,
            #     #nonzero=True,
            #     channel_wise=True,
            # ),
            # ScaleIntensityRanged(
            #     keys=["image"],
            #     b_min=0,
            #     b_max=1,
            # ),
            # ThresholdMaskingd(
            #     keys=["image"],
            # ),
        ]
    )
    val_transforms.set_random_state(seed=0)
    return train_transforms, val_transforms, test_transforms

def set_deterministic_mode(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
