import torch
import numpy as np
from monai.transforms import Transform
from typing import Dict, Tuple, List
import torch.fft
from copy import deepcopy
import random


class CropAndStackTransformd(Transform):
    """
    A MONAI-like dictionary-based transform that takes an image tensor of shape (1, 1, 1024, 1024)
    and crops it into four quadrants, then stacks them along a new channel dimension.
    """

    def __init__(self, keys):
        self.keys = keys
    
    def __call__(self, data):
        d = dict(data)
        
        for key in self.keys:
            image = d[key]  # Expected shape: (1, 1, 1024, 1024)
            
            # Define crop coordinates for four quadrants
            crops = [
                (0, 512, 0, 512),    # Top-left
                (0, 512, 512, 1024), # Top-right
                (512, 1024, 0, 512), # Bottom-left
                (512, 1024, 512, 1024) # Bottom-right
            ]
            
            # Extract crops and stack them along a new dimension
            cropped_images = []
            random_crops = random.sample(crops, len(crops))
            for (y1, y2, x1, x2) in random_crops:
                crop = image[:, y1:y2, x1:x2]  # Crop shape: (1, 1, 512, 512)
                cropped_images.append(crop)
            
            # Stack crops along a new channel dimension
            d[key] = torch.cat(cropped_images, dim=0)  # Final shape: (1, 4, 512, 512)
            
        return d


class MIPTransformd(Transform):
    """
    A MONAI-like dictionary-based transform that takes an image tensor of shape (1, 1, 1024, 1024)
    and crops it into four quadrants. It then computes the maximum intensity projection 
    across the cropped quadrants, resulting in a single (1, 512, 512) image.
    """

    def __init__(self, keys):
        self.keys = keys
        self.crops = [
            (0, 512, 0, 512),    # Top-left
            (0, 512, 512, 1024), # Top-right
            (512, 1024, 0, 512), # Bottom-left
            (512, 1024, 512, 1024) # Bottom-right
        ]

    def __call__(self, data):
        d = dict(data)
        
        for key in self.keys:
            image = d[key]  # Expected shape: (1, 1, 1024, 1024)
            
            # Randomize the order of crops if desired
            random_crops = random.sample(self.crops, len(self.crops))
            
            # Extract the crops
            cropped_images = []
            for (y1, y2, x1, x2) in random_crops:
                crop = image[:, y1:y2, x1:x2]  # Crop shape: (1, 1, 512, 512)
                cropped_images.append(crop)
            
            # Stack crops along a new dimension (batch dimension) and compute max intensity projection
            stacked_crops = torch.cat(cropped_images, dim=0)  # Shape: (4, 1, 512, 512)
            mip_image = torch.max(stacked_crops, dim=0).values.unsqueeze(0)
            #print(mip_image.shape)
            # Update the dictionary with the MIP image
            d[key] = mip_image
        
        return d
        
class RandomSpatialMasking(Transform):
    def __init__(self, patch_size: int, mask_fraction: float, shift_range: int):
        """
        Initialize the SpatialMaskingTransform.
        
        Args:
            patch_size (int): Size of the patches used in the mask.
            mask_fraction (float): Fraction of the image to be masked (e.g., 0.25 for 25%).
            shift_range (int): Maximum range for random shift applied to the mask.
        """
        self.patch_size = patch_size
        self.mask_fraction = mask_fraction
        self.shift_range = shift_range
    
    def create_random_patch_mask(self, H: int, W: int, patch_size: int, num_patches: int) -> torch.Tensor:
        """
        Create a random patch mask of the given dimensions.
        
        Args:
            H (int): Height of the image.
            W (int): Width of the image.
            patch_size (int): Size of each patch.
            num_patches (int): Number of patches in the mask.
        
        Returns:
            torch.Tensor: Random patch mask with the same spatial dimensions as the image.
        """
        mask = torch.ones(H, W)
        for _ in range(num_patches):
            top = np.random.randint(0, H - patch_size)
            left = np.random.randint(0, W - patch_size)
            mask[top:top + patch_size, left:left + patch_size] = 0
        return mask

    def apply_random_shift(self, mask: torch.Tensor, shift_range: int) -> torch.Tensor:
        """
        Apply a random spatial shift to the mask.
        
        Args:
            mask (torch.Tensor): The original mask.
            shift_range (int): Maximum range for the shift.
        
        Returns:
            torch.Tensor: Mask with applied random shift.
        """
        shift_y = np.random.randint(-shift_range, shift_range + 1)
        shift_x = np.random.randint(-shift_range, shift_range + 1)
        shifted_mask = torch.roll(mask, shifts=(shift_y, shift_x), dims=(0, 1))
        return shifted_mask

    def __call__(self, images: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply the spatial masking transformation to a list of images.
        
        Args:
            images (List[torch.Tensor]): List containing image tensors to be masked.
        
        Returns:
            List[torch.Tensor]: List containing masked images.
        """
        if not images:
            return images
        
        B, C, H, W = images[0].shape
        device = images[0].device
        
        # Compute number of patches needed
        patch_area = self.patch_size * self.patch_size
        total_area = H * W
        num_patches = int(np.ceil(self.mask_fraction * total_area / patch_area))

        # Generate random patch-based mask for the entire batch
        base_mask = self.create_random_patch_mask(H, W, self.patch_size, num_patches)
        # Apply a unique shift to the mask for each image in the batch
        masked_images = [img.clone().to(device) for img in images]

        for i in range(B):
            for j in range(len(images)):
                shifted_mask = self.apply_random_shift(base_mask, self.shift_range)
                shifted_mask = shifted_mask.unsqueeze(0).expand(C, -1, -1)  # Expand mask to match number of channels
                masked_images[j][i] = (images[j][i] * shifted_mask.to(device))
        
        return masked_images


class BalancedSpectralMasking(Transform):
    def __init__(self, num_bands=64):
        self.num_bands = num_bands

    def __call__(self, images: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply the spectral masking transformation to a list of images.

        Args:
            images (List[torch.Tensor]): List containing image tensors to be masked.

        Returns:
            List[torch.Tensor]: List containing masked images.
        """
        if not images:
            return images

        B, C, H, W = images[0].shape
        device = images[0].device

        masked_images = [img.clone().to(device) for img in images]

        for i in range(B):
            for j in range(len(images)):
                image = images[j][i].to(device)
                img_fft = torch.fft.fft2(image)

                img_fft_shift = torch.fft.fftshift(img_fft)
                u = torch.fft.fftfreq(H, device=device)
                v = torch.fft.fftfreq(W, device=device)
                u, v = torch.meshgrid(u, v, indexing="xy")
                u = torch.fft.fftshift(u)
                v = torch.fft.fftshift(v)
                freq_magnitude = torch.sqrt(u ** 2 + v ** 2)

                freq_magnitude = freq_magnitude / freq_magnitude.max() + torch.tensor(1e-10, device=device)

                band_edges = torch.linspace(0, 1, self.num_bands + 1, device=device)
                band_masks = torch.zeros((self.num_bands, H, W), dtype=torch.bool, device=device)

                for k in range(self.num_bands):
                    band_masks[k] = (freq_magnitude >= band_edges[k]) & (freq_magnitude < band_edges[k + 1])

                spectral_bands = torch.zeros((self.num_bands, C, H, W), dtype=torch.complex64, device=device)
                specs = torch.zeros((self.num_bands, C, H, W), device=device)
                for k in range(self.num_bands):
                    band_fft = img_fft_shift * band_masks[k][None, ...]
                    spectral_bands[k] = band_fft
                    specs[k] = torch.real(torch.fft.ifft2(torch.fft.ifftshift(band_fft))).type(torch.float32)

                content_sum = torch.sum(torch.abs(specs), dim=(2, 3))
                content_sum_sum = torch.sum(content_sum)
                normalized_content = content_sum / content_sum_sum

                # Step 3: Generate balanced spectral mask
                mask_probs = normalized_content
                random_values = torch.rand(self.num_bands, device=device)
                mask = [0 if rand_num <= prob else 1 for rand_num, prob in zip(random_values, mask_probs.squeeze())]
                
                #print(random_values, mask, torch.flip(mask_probs, dims=(0,)).squeeze())
                # Step 4: Apply mask
                reconstructed_fft_shift = torch.zeros_like(img_fft_shift)
                for k in range(self.num_bands):
                    band_fft = img_fft_shift * band_masks[k][None, ...] * mask[k]
                    reconstructed_fft_shift += band_fft
                    
                #reconstructed_fft = torch.fft.ifftshift(reconstructed_fft_shift)
                masked_image = torch.real(torch.fft.ifft2(torch.fft.ifftshift(reconstructed_fft_shift)))
                masked_image = torch.clamp(masked_image, min=0)
                if torch.isnan(masked_image).any() or torch.isinf(masked_image).any():
                    masked_image = image
                masked_images[j][i] = masked_image
        return masked_images


class ThresholdMaskingd(Transform):
    def __init__(self, keys: list, threshold=0.98):
        """
        Initialize the transform.

        Args:
            key (str): The dictionary key where the images are stored.
            num_bands (int): Not used here but can be adapted for more advanced masking.
            threshold (float): The threshold value for masking.
        """
        self.keys = keys
        self.threshold = threshold

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply the spectral masking transformation to the images stored under the key.

        Args:
            data (Dict[str, torch.Tensor]): Input data dictionary containing images.

        Returns:
            Dict[str, torch.Tensor]: Updated dictionary with images including an additional channel.
        """
        
        for key in self.keys:
            images = data[key]
        
            if not isinstance(images, torch.Tensor) or images.ndim != 3:
                raise ValueError(f"Expected 3D tensor (C, H, W) for key '{key}', got {images.shape}")
            
            C, H, W = images.shape
            device = images.device
    
            masked_images = images.clone().to(device)  # Keep original images
            
            new_channels = []
    
            
            for j in range(C):  # Process each channel individually
                image = images[j].to(device)
                
                # FFT
                img_fft = torch.fft.fft2(image)
                img_fft_shift = torch.fft.fftshift(img_fft)

                magnitude_spectrum_with_dc = torch.log(torch.abs(img_fft_shift))
                
                # Remove DC component
                center_h, center_w = H // 2, W // 2
                img_fft_shift_no_dc = img_fft_shift.clone()
                img_fft_shift_no_dc[center_h, center_w] = 0
                
                # Apply thresholding
                threshold = torch.log(torch.abs(img_fft_shift_no_dc)).max() * self.threshold
                groove_mask = (magnitude_spectrum_with_dc <= threshold).float()

                magnitude = torch.abs(img_fft_shift)
                phase = torch.angle(img_fft_shift)
                
                # Masking
                masked_magnitude = magnitude * groove_mask
                img_fft_shift_masked = masked_magnitude * torch.exp(1j * phase)
                
                # Inverse FFT
                masked_im_no_dc = torch.real(torch.fft.ifft2(torch.fft.ifftshift(img_fft_shift_masked)))
                
                # Append the new channel (masked image)
                new_channels.append(masked_im_no_dc.unsqueeze(0))  # Keep batch dimension
                    
            # Concatenate new channel with original channels
            new_channel_tensor = torch.cat(new_channels)  # (B, C_new, H, W)
            data[key] = torch.cat([masked_images, new_channel_tensor], dim=0)  # Add new channels to original images
            
        return data