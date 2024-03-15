import torch
from monai.transforms import Compose, RandRotated, RandRotate90d, Resized, RandZoomd, RandAdjustContrastd, RandCropByPosNegLabeld, NormalizeIntensityd, RandFlipd


def compute_mean_std(dataset):
    """
    Compute the mean and standard deviation for a given dataset.

    Args:
    - dataset: PyTorch dataset containing images

    Returns:
    - mean: Mean value of the dataset
    - std: Standard deviation of the dataset
    """
    mean = torch.zeros(1)
    std = torch.zeros(1)
    for i in range(len(dataset)):
        image = dataset[i]["image"]
        mean += torch.mean(image)
        std += torch.std(image)

    mean /= len(dataset)
    std /= len(dataset)

    # Convert to numpy arrays for easier handling
    mean = mean.numpy()
    std = std.numpy()

    return mean, std


def create_transforms():
    train_transforms = Compose(
        [
            Resized(
                keys=["image"],
                spatial_size=(512, 512),
                mode="bilinear",
            ),
            # RandSpatialCropSamplesd(
            #     keys=["image"],
            #     roi_size=(512, 512),
            #     num_samples=4,
            # ),
            # RandCropByPosNegLabeld(
            #     keys=["image"],
            #     spatial_size=(512, 512),
            #     num_samples=4,
            #     label_key="image",
            #     pos=10,
            #     neg=0,
            #     image_key="image",
            #     image_threshold=0.1,
            # ),

            NormalizeIntensityd(
                keys=["image"],
                # subtrahend=train_mean,
                # divisor=train_std,
            ),
            RandFlipd(
                keys=["image"],
                prob=0.7,
                spatial_axis=1,
            ),
            RandFlipd(
                keys=["image"],
                prob=0.7,
                spatial_axis=0,
            ),
            RandRotated(
                keys=["image"],
                range_x=3.14,
                prob=0.99,
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
                # subtrahend=train_mean,
                # divisor=train_std,
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
                # subtrahend=train_mean,
                # divisor=train_std,
            ),
        ]
    )

    return train_transforms, val_transforms, test_transforms
