import torch
import einops
import numpy as np
import torch.nn as nn
from torch import Tensor
from functools import partial
from torchdiffeq import odeint

from unet import UNetModel
from diffusers import AutoencoderKL


def exists(val):
    return val is not None


class DepthFM(nn.Module):
    def __init__(self, ckpt_path: str):
        super().__init__()
        vae_id = "runwayml/stable-diffusion-v1-5"
        self.vae = AutoencoderKL.from_pretrained(vae_id, subfolder="vae")
        self.scale_factor = 0.18215

        # set with checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.noising_step = ckpt['noising_step']
        self.empty_text_embed = ckpt['empty_text_embedding']
        self.model = UNetModel(**ckpt['ldm_hparams'])
        self.model.load_state_dict(ckpt['state_dict'])
    
    def ode_fn(self, t: Tensor, x: Tensor, **kwargs):
        if t.numel() == 1:
            t = t.expand(x.size(0))
        return self.model(x=x, t=t, **kwargs)
    
    def generate(self, z: Tensor, num_steps: int = 12, n_intermediates: int = 0, **kwargs):
        """
        ODE solving from z0 (ims) to z1 (depth).
        """
        ode_kwargs = dict(method="euler", rtol=1e-5, atol=1e-5, options=dict(step_size=1.0 / num_steps))
        
        # t specifies which intermediate times should the solver return
        # e.g. t = [0, 0.5, 1] means return the solution at t=0, t=0.5 and t=1
        # but it also specifies the number of steps for fixed step size methods
        t = torch.linspace(0, 1, n_intermediates + 2, device=z.device, dtype=z.dtype)
        # t = torch.tensor([0., 1.], device=z.device, dtype=z.dtype)

        # allow conditioning information for model
        ode_fn = partial(self.ode_fn, **kwargs)
        
        ode_results = odeint(ode_fn, z, t, **ode_kwargs)
        
        if n_intermediates > 0:
            return ode_results
        return ode_results[-1]
    
    def forward(self, ims: Tensor, num_steps: int = 8, ensemble_size: int = 4):
        """
        Args:
            ims: Tensor of shape (b, 3, h, w) in range [-1, 1]
        Returns:
            depth: Tensor of shape (b, 1, h, w) in range [0, 1]
        """
        if ensemble_size > 1:
            assert ims.shape[0] == 1, "Ensemble mode only supported with batch size 1"
            ims = ims.repeat(ensemble_size, 1, 1, 1)
        
        bs, dev = ims.shape[0], ims.device

        ims_z = self.encode(ims, sample_posterior=False)

        conditioning = torch.tensor(self.empty_text_embed).to(dev).repeat(bs, 1, 1)
        context = ims_z
        
        x_source = ims_z

        if self.noising_step > 0:
            x_source = q_sample(x_source, self.noising_step)    

        # solve ODE
        depth_z = self.generate(x_source, num_steps=num_steps, context=context, context_ca=conditioning)

        depth = self.decode(depth_z)
        depth = depth.mean(dim=1, keepdim=True)

        if ensemble_size > 1:
            depth = depth.mean(dim=0, keepdim=True)
        
        # normalize depth maps to range [-1, 1]
        depth = per_sample_min_max_normalization(depth.exp())


        return depth
    
    @torch.no_grad()
    def predict_depth(self, ims: Tensor, num_steps: int = 4, ensemble_size: int = 1):
        """ Inference method for DepthFM. """
        return self.forward(ims, num_steps, ensemble_size)
    
    @torch.no_grad()
    def encode(self, x: Tensor, sample_posterior: bool = True):
        posterior = self.vae.encode(x)
        if sample_posterior:
            z = posterior.latent_dist.sample()
        else:
            z = posterior.latent_dist.mode()
        # normalize latent code
        z = z * self.scale_factor
        return z
    
    @torch.no_grad()
    def decode(self, z: Tensor):
        z = 1.0 / self.scale_factor * z
        return self.vae.decode(z).sample


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def cosine_log_snr(t, eps=0.00001):
    """
    Returns log Signal-to-Noise ratio for time step t and image size 64
    eps: avoid division by zero
    """
    return -2 * np.log(np.tan((np.pi * t) / 2) + eps)


def cosine_alpha_bar(t):
    return sigmoid(cosine_log_snr(t))


def q_sample(x_start: torch.Tensor, t: int, noise: torch.Tensor = None, n_diffusion_timesteps: int = 1000):
    """
    Diffuse the data for a given number of diffusion steps. In other
    words sample from q(x_t | x_0).
    """
    dev = x_start.device
    dtype = x_start.dtype

    if noise is None:
        noise = torch.randn_like(x_start)
    
    alpha_bar_t = cosine_alpha_bar(t / n_diffusion_timesteps)
    alpha_bar_t = torch.tensor(alpha_bar_t).to(dev).to(dtype)

    return torch.sqrt(alpha_bar_t) * x_start + torch.sqrt(1 - alpha_bar_t) * noise


def per_sample_min_max_normalization(x):
    """ Normalize each sample in a batch independently
    with min-max normalization to [0, 1] """
    bs, *shape = x.shape
    x_ = einops.rearrange(x, "b ... -> b (...)")
    min_val = einops.reduce(x_, "b ... -> b", "min")[..., None]
    max_val = einops.reduce(x_, "b ... -> b", "max")[..., None]
    x_ = (x_ - min_val) / (max_val - min_val)
    return x_.reshape(bs, *shape)
    
    
import torch
import numpy as np

def compute_metrics(pred_depth: torch.Tensor, gt_depth: torch.Tensor, mask: torch.Tensor = None) -> dict:
    """
    Compute comprehensive depth estimation metrics including:
    - RMSE, REL, δ1-δ3 (standard metrics)
    - Accuracy (pixel-wise correctness within threshold)
    - RBSREL (Robust Scale-Invariant Relative Error)
    """
    if mask is None:
        mask = gt_depth > 0  # Assume invalid pixels are 0
    
    # Apply mask and ensure numerical stability
    pred_depth = pred_depth * mask
    gt_depth = gt_depth * mask
    eps = 1e-6

    # Standard metrics
    rmse = torch.sqrt(torch.mean((pred_depth - gt_depth) ** 2))
    rel = torch.mean(torch.abs(pred_depth - gt_depth) / (gt_depth + eps))
    
    # Threshold metrics (δ1, δ2, δ3)
    threshold = torch.max(
        (pred_depth + eps) / (gt_depth + eps),
        (gt_depth + eps) / (pred_depth + eps)
    )
    delta1 = torch.mean((threshold < 1.25).float()) * 100
    delta2 = torch.mean((threshold < 1.25**2).float()) * 100
    delta3 = torch.mean((threshold < 1.25**3).float()) * 100

    # Accuracy (percentage of pixels within absolute threshold)
    abs_diff = torch.abs(pred_depth - gt_depth)
    accuracy_5cm = torch.mean((abs_diff < 0.05).float()) * 100  # 5cm threshold
    accuracy_10cm = torch.mean((abs_diff < 0.10).float()) * 100  # 10cm threshold

    # RBSREL (Robust Scale-Invariant Relative Error)
    log_diff = torch.log(pred_depth + eps) - torch.log(gt_depth + eps)
    rbsrel = torch.mean(torch.abs(log_diff)) * 100  # Percentage

    return {
        "RMSE": rmse.item(),
        "REL": rel.item(),
        "δ1": delta1.item(),
        "δ2": delta2.item(),
        "δ3": delta3.item(),
        "Accuracy@5cm": accuracy_5cm.item(),
        "Accuracy@10cm": accuracy_10cm.item(),
        "RBSREL": rbsrel.item()
    }
def generate(self, z: Tensor, num_steps: int = 4, n_intermediates: int = 0, **kwargs):
    """
    ODE solving from z0 (ims) to z1 (depth) with improved stability.
    """
    # Use adaptive step size solver for better stability
    ode_kwargs = dict(
        method="dopri5",  # More stable adaptive solver
        rtol=1e-7,
        atol=1e-5,
        options=dict(
            max_num_steps=num_steps * 100,  # Allow more steps for convergence
            first_step=0.05,  # Initial step size
            safety=0.9,  # Safety factor for step size adaptation
        )
    )
    
    t = torch.linspace(0, 1, n_intermediates + 2, device=z.device, dtype=z.dtype)
    ode_fn = partial(self.ode_fn, **kwargs)
    
    try:
        ode_results = odeint(ode_fn, z, t, **ode_kwargs)
    except AssertionError as e:
        if "underflow in dt" in str(e):
            # Fallback to fixed-step solver if adaptive fails
            print("Warning: Adaptive solver failed, switching to fixed-step Euler")
            ode_kwargs = dict(
                method="euler",
                options=dict(step_size=1.0 / num_steps)
            )
            ode_results = odeint(ode_fn, z, t, **ode_kwargs)
        else:
            raise
    
    return ode_results[-1] if n_intermediates == 0 else ode_results
def forward(self, ims: torch.Tensor, gt_depth: torch.Tensor = None, 
            num_steps: int = 4, ensemble_size: int = 1):
    """
    Forward pass with enhanced metric reporting
    """
    depth = self.predict_depth(ims, num_steps=num_steps, ensemble_size=ensemble_size)
    
    if gt_depth is not None:
        # Ensure GT matches prediction size
        if gt_depth.shape[-2:] != depth.shape[-2:]:
            gt_depth = F.interpolate(gt_depth, size=depth.shape[-2:], mode='bilinear')
        
        metrics = compute_metrics(depth, gt_depth)
        return depth, metrics
    
    return depth
