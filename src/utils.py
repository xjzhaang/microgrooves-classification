import torch
import numpy as np
from monai.transforms import Compose, RandRotated, RandRotate90d, Resized, RandZoomd, RandAdjustContrastd, RandCropByPosNegLabeld, NormalizeIntensityd, RandFlipd, RandShiftIntensityd

# def compute_mean_std(dataset):
#     """
#     Compute the mean and standard deviation for a given dataset.

#     Args:
#     - dataset: PyTorch dataset containing images

#     Returns:
#     - mean: Mean value of the dataset
#     - std: Standard deviation of the dataset
#     """
#     mean = torch.zeros(1)
#     std = torch.zeros(1)
#     for i in range(len(dataset)):
#         image = dataset[i]["image"]
#         mean += torch.mean(image)
#         std += torch.std(image)
#     mean /= len(dataset)
#     std /= len(dataset)

#     # Convert to numpy arrays for easier handling
#     mean = mean.numpy()
#     std = std.numpy()

#     return mean, std


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
                keys=["image"],
                spatial_size=(512, 512),
                mode="bilinear",
            ),
            NormalizeIntensityd(
                keys=["image"],
                # subtrahend=mean,
                # divisor=std,
                channel_wise=True,
            ),
            RandFlipd(
                keys=["image"],
                prob=0.5,
                spatial_axis=1,
            ),
            RandFlipd(
                keys=["image"],
                prob=0.5,
                spatial_axis=0,
            ),
            RandRotated(
                keys=["image"],
                range_x=3.14,
                prob=1,
                padding_mode="zeros",
            ),
            RandRotate90d(
                keys=["image"],
                prob=0.5,
                max_k=3,
            ),
            RandZoomd(
                keys=["image"],
                min_zoom=0.9,
                max_zoom=1.1,
                prob=0.3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                prob=0.2,
                offsets=0.2,
            ),
            RandAdjustContrastd(
                keys=["image"],
                prob=0.2,
                gamma=(0.5, 3),
            )
        ]
    )

    val_transforms = Compose(
        [
            Resized(
                keys=["image"],
                spatial_size=(512, 512),
                mode="bilinear",
            ),
            NormalizeIntensityd(
                keys=["image"],
                # subtrahend=mean,
                # divisor=std,
                channel_wise=True,
            ),
        ]
    )
    test_transforms = Compose(
        [
            Resized(
                keys=["image"],
                spatial_size=(1024, 1024),
                mode="bilinear",
            ),
            # Resized(
            #     keys=["image"],
            #     spatial_size=(512, 512),
            #     mode="bilinear",
            # ),
            NormalizeIntensityd(
                keys=["image"],
                # subtrahend=mean,
                # divisor=std,
                channel_wise=True,
            ),
        ]
    )

    return train_transforms, val_transforms, test_transforms
