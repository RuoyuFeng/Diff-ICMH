import importlib
from inspect import isfunction
from typing import Optional, Tuple, Union

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch import optim

from .common import frozen_module
from .image import rgb2ycbcr_pt


# https://github.com/XPixelGroup/BasicSR/blob/033cd6896d898fdd3dcda32e3102a792efa1b8f4/basicsr/metrics/psnr_ssim.py#L52
def calculate_psnr_pt(img, img2, crop_border, test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) (PyTorch version).

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: PSNR result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    mse = torch.mean((img - img2)**2, dim=[1, 2, 3])
    return 10. * torch.log10(1. / (mse + 1e-8))


class LPIPS:
    
    def __init__(self, net: str) -> None:
        self.model = lpips.LPIPS(net=net)
        frozen_module(self.model)
    
    @torch.no_grad()
    def __call__(self, img1: torch.Tensor, img2: torch.Tensor, normalize: bool) -> torch.Tensor:
        """
        Compute LPIPS.
        
        Args:
            img1 (torch.Tensor): The first image (NCHW, RGB, [-1, 1]). Specify `normalize` if input 
                image is range in [0, 1].
            img2 (torch.Tensor): The second image (NCHW, RGB, [-1, 1]). Specify `normalize` if input 
                image is range in [0, 1].
            normalize (bool): If specified, the input images will be normalized from [0, 1] to [-1, 1].
            
        Returns:
            lpips_values (torch.Tensor): The lpips scores of this batch.
        """
        return self.model(img1, img2, normalize=normalize)
    
    def to(self, device: str) -> "LPIPS":
        self.model.to(device)
        return self


def compute_psnr( x, y):
    EPS = 1e-8
    mse = torch.mean((x - y) ** 2, dim=[1, 2, 3])
    psnr = - 10 * torch.log10(mse + EPS)
    return psnr.mean(dim=0)


def compute_ssim( x, y):
        kernel_size = 11
        kernel_sigma = 1.5
        k1 = 0.01
        k2 = 0.03
        
        f = max(1, round(min(x.size()[-2:]) / 256))
        if (f > 1) :
            x = F.avg_pool2d(x, kernel_size=f)
            y = F.avg_pool2d(y, kernel_size=f)
            
        kernel = gaussian_filter(kernel_size, kernel_sigma, device=x.device, dtype=x.dtype).repeat(x.size(1), 1, 1, 1)
        _compute_ssim_per_channel = _ssim_per_channel_complex if x.dim() == 5 else _ssim_per_channel
        ssim_map, cs_map = _compute_ssim_per_channel(x=x, y=y, kernel=kernel, data_range=1, k1=k1, k2=k2)
        ssim_val = ssim_map.mean(1)
        return ssim_val.mean(dim=0)       
    
    
def _ssim_per_channel(x: torch.Tensor, y: torch.Tensor, kernel: torch.Tensor,
                      data_range: Union[float, int] = 1., k1: float = 0.01,
                      k2: float = 0.03) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Calculate Structural Similarity (SSIM) index for X and Y per channel.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        kernel: 2D Gaussian kernel.
        data_range: Maximum value range of images (usually 1.0 or 255).
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Full Value of Structural Similarity (SSIM) index.
    """
    if x.size(-1) < kernel.size(-1) or x.size(-2) < kernel.size(-2):
        raise ValueError(f'Kernel size can\'t be greater than actual input size. '
                         f'Input size: {x.size()}. Kernel size: {kernel.size()}')

    c1 = k1 ** 2
    c2 = k2 ** 2
    n_channels = x.size(1)
    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_xx = F.conv2d(x ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xx
    sigma_yy = F.conv2d(y ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_yy
    sigma_xy = F.conv2d(x * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xy

    # Contrast sensitivity (CS) with alpha = beta = gamma = 1.
    cs = (2. * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)

    # Structural similarity (SSIM)
    ss = (2. * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs

    ssim_val = ss.mean(dim=(-1, -2))
    cs = cs.mean(dim=(-1, -2))
    return ssim_val, cs


def _ssim_per_channel_complex(x: torch.Tensor, y: torch.Tensor, kernel: torch.Tensor,
                              data_range: Union[float, int] = 1., k1: float = 0.01,
                              k2: float = 0.03) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Calculate Structural Similarity (SSIM) index for Complex X and Y per channel.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W, 2)`.
        y: A target tensor. Shape :math:`(N, C, H, W, 2)`.
        kernel: 2-D gauss kernel.
        data_range: Maximum value range of images (usually 1.0 or 255).
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Full Value of Complex Structural Similarity (SSIM) index.
    """
    n_channels = x.size(1)
    if x.size(-2) < kernel.size(-1) or x.size(-3) < kernel.size(-2):
        raise ValueError(f'Kernel size can\'t be greater than actual input size. Input size: {x.size()}. '
                         f'Kernel size: {kernel.size()}')

    c1 = k1 ** 2
    c2 = k2 ** 2

    x_real = x[..., 0]
    x_imag = x[..., 1]
    y_real = y[..., 0]
    y_imag = y[..., 1]

    mu1_real = F.conv2d(x_real, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu1_imag = F.conv2d(x_imag, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu2_real = F.conv2d(y_real, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu2_imag = F.conv2d(y_imag, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu1_sq = mu1_real.pow(2) + mu1_imag.pow(2)
    mu2_sq = mu2_real.pow(2) + mu2_imag.pow(2)
    mu1_mu2_real = mu1_real * mu2_real - mu1_imag * mu2_imag
    mu1_mu2_imag = mu1_real * mu2_imag + mu1_imag * mu2_real

    compensation = 1.0

    x_sq = x_real.pow(2) + x_imag.pow(2)
    y_sq = y_real.pow(2) + y_imag.pow(2)
    x_y_real = x_real * y_real - x_imag * y_imag
    x_y_imag = x_real * y_imag + x_imag * y_real

    sigma1_sq = F.conv2d(x_sq, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_sq
    sigma2_sq = F.conv2d(y_sq, weight=kernel, stride=1, padding=0, groups=n_channels) - mu2_sq
    sigma12_real = F.conv2d(x_y_real, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_mu2_real
    sigma12_imag = F.conv2d(x_y_imag, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_mu2_imag
    sigma12 = torch.stack((sigma12_imag, sigma12_real), dim=-1)
    mu1_mu2 = torch.stack((mu1_mu2_real, mu1_mu2_imag), dim=-1)
    # Set alpha = beta = gamma = 1.
    cs_map = (sigma12 * 2 + c2 * compensation) / (sigma1_sq.unsqueeze(-1) + sigma2_sq.unsqueeze(-1) + c2 * compensation)
    ssim_map = (mu1_mu2 * 2 + c1 * compensation) / (mu1_sq.unsqueeze(-1) + mu2_sq.unsqueeze(-1) + c1 * compensation)
    ssim_map = ssim_map * cs_map

    ssim_val = ssim_map.mean(dim=(-2, -3))
    cs = cs_map.mean(dim=(-2, -3))

    return ssim_val, cs


def gaussian_filter(kernel_size: int, sigma: float, device: Optional[str] = None,
                    dtype: Optional[type] = None) -> torch.Tensor:
    r"""Returns 2D Gaussian kernel N(0,`sigma`^2)
    Args:
        size: Size of the kernel
        sigma: Std of the distribution
        device: target device for kernel generation
        dtype: target data type for kernel generation
    Returns:
        gaussian_kernel: Tensor with shape (1, kernel_size, kernel_size)
    """
    coords = torch.arange(kernel_size, dtype=dtype, device=device)
    coords -= (kernel_size - 1) / 2.

    g = coords ** 2
    g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma ** 2)).exp()

    g /= g.sum()
    return g.unsqueeze(0)

#! Attention! Written via Claude3.7. Not test yet.
def compute_msssim(x, y, weights=None, data_range=1.0):
    """
    Compute Multi-Scale Structural Similarity Index (MS-SSIM).
    
    Args:
        x (torch.Tensor): First image batch in range [0, 1], shape (N, C, H, W).
        y (torch.Tensor): Second image batch in range [0, 1], shape (N, C, H, W).
        weights (torch.Tensor, optional): Weights for different scales. Default weights follow the paper.
        data_range (float): The data range of the input image (usually 1.0 or 255).
        
    Returns:
        torch.Tensor: MS-SSIM score between x and y (average over batch).
    """
    if weights is None:
        # Default weights from the MS-SSIM paper
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(x.device)
    
    # Number of scales is determined by weights
    levels = weights.size(0)
    
    # Check minimum image size for multi-scale processing
    min_size = 2**(levels-1) + 1  # Minimum size for the smallest scale
    if x.size(-1) < min_size or x.size(-2) < min_size:
        # If the image is too small for all scales, adjust levels
        max_level = int(np.log2(min(x.size(-1), x.size(-2)))) - 1
        levels = max(1, max_level)
        weights = weights[:levels]
        weights = weights / weights.sum()
    
    msssim = []
    mcs = []
    
    # Parameters for SSIM calculation
    kernel_size = 11
    kernel_sigma = 1.5
    k1 = 0.01
    k2 = 0.03
    kernel = gaussian_filter(kernel_size, kernel_sigma, device=x.device, dtype=x.dtype).repeat(x.size(1), 1, 1, 1)
    
    for i in range(levels):
        # Calculate SSIM and contrast on current scale
        ssim_map, cs_map = _ssim_per_channel(
            x=x, y=y, kernel=kernel, data_range=data_range, k1=k1, k2=k2
        )
        
        # Store results
        msssim.append(ssim_map)
        mcs.append(cs_map)
        
        # Skip downsampling for the last scale
        if i < levels - 1:
            # Downsample for next scale
            padding = (x.shape[2] % 2, x.shape[3] % 2)
            x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=padding)
            y = F.avg_pool2d(y, kernel_size=2, stride=2, padding=padding)
    
    # Convert lists to tensors
    msssim = torch.stack(msssim, dim=0)  # (levels, batch, channels)
    mcs = torch.stack(mcs, dim=0)        # (levels, batch, channels)
    
    # Calculate weighted combination:
    # Use the original MS-SSIM formula: prod(mcs[i]^weights[i]) * msssim[levels-1]^weights[levels-1]
    # For numerical stability, convert to log domain then back
    mcs_power = mcs ** weights.view(-1, 1, 1)
    msssim_power = msssim[-1:] ** weights.view(-1, 1, 1)[-1:]
    
    # Take product of all levels (using log for numerical stability)
    mcs_log = torch.log(torch.clamp(mcs_power, min=1e-8))
    msssim_log = torch.log(torch.clamp(msssim_power, min=1e-8))
    
    # Sum in log domain (equivalent to product)
    mcs_sum = torch.sum(mcs_log[:-1], dim=0)
    msssim_sum = msssim_log.squeeze(0)
    
    # Combine mcs and last level ssim
    ms_ssim_val = torch.exp(mcs_sum + msssim_sum)
    
    # Average over channels and batch
    return ms_ssim_val.mean(dim=1).mean(dim=0)