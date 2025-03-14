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


class RandomFrequencyMasking(torch.nn.Module):
    def __init__(
        self, 
        min_threshold: float = 0.8, 
        max_threshold: float = 0.9999,
        p: float = 0.3
    ):
        """
        Initialize the transform.
        Args:
            min_threshold (float): Minimum threshold value for masking.
            max_threshold (float): Maximum threshold value for masking.
            p (float): Probability of applying this transform.
        """
        super().__init__()
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.p = p
        
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply the spectral masking transformation to the input image.
        Args:
            img (torch.Tensor): Input image tensor of shape (C, H, W).
        Returns:
            torch.Tensor: Transformed image with same shape as input.
        """
        # Skip the transform with probability 1-p
        if random.random() > self.p:
            return img
            
        # Randomly sample threshold for this batch
        threshold = random.uniform(self.min_threshold, self.max_threshold)
        
        if not isinstance(img, torch.Tensor) or img.ndim != 3:
            raise ValueError(f"Expected 3D tensor (C, H, W), got {img.shape}")
        
        C, H, W = img.shape
        device = img.device
        
        result = torch.zeros_like(img)

        for j in range(C):  # Process each channel individually
            image = img[j].to(device)
            
            # FFT
            img_fft = torch.fft.fft2(image)
            img_fft_shift = torch.fft.fftshift(img_fft)
            magnitude_spectrum_with_dc = torch.log(torch.abs(img_fft_shift))
            
            # Remove DC component
            center_h, center_w = H // 2, W // 2
            img_fft_shift_no_dc = img_fft_shift.clone()
            img_fft_shift_no_dc[center_h, center_w] = 0
            
            # Apply thresholding
            threshold_value = torch.log(torch.abs(img_fft_shift_no_dc)).max() * threshold
            groove_mask = (magnitude_spectrum_with_dc <= threshold_value).float()
            magnitude = torch.abs(img_fft_shift)
            phase = torch.angle(img_fft_shift)
            
            # Masking
            masked_magnitude = magnitude * groove_mask
            img_fft_shift_masked = masked_magnitude * torch.exp(1j * phase)
            
            # Inverse FFT
            masked_im_no_dc = torch.real(torch.fft.ifft2(torch.fft.ifftshift(img_fft_shift_masked)))
            
            # Store the masked image in the result tensor
            result[j] = masked_im_no_dc
                
        return result