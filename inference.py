#!/usr/bin/env python3
import os
import argparse
import torch
import numpy as np
from PIL import Image, ImageOps
from depthfm import DepthFM
import matplotlib.pyplot as plt
from einops import rearrange
from typing import Optional, Tuple, Dict, List
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def get_dtype_from_str(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype"""
    return {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16
    }[dtype_str]

def resize_max_res(
    img: Image.Image, 
    max_resolution: int, 
    resample_method=Image.BILINEAR
) -> Tuple[Image.Image, Tuple[int, int]]:
    """Resize image while maintaining aspect ratio"""
    orig_w, orig_h = img.size
    scale = min(max_resolution / orig_w, max_resolution / orig_h)
    
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    new_w = (new_w // 64) * 64
    new_h = (new_h // 64) * 64
    
    print(f"Resized: {orig_w}x{orig_h} → {new_w}x{new_h}")
    return img.resize((new_w, new_h), resample_method), (orig_w, orig_h)

def load_image(
    filepath: str, 
    processing_res: int = -1,
    is_depth: bool = False
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Load and preprocess image or depth map"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image not found: {filepath}")
    
    img = Image.open(filepath)
    if is_depth:
        img = ImageOps.grayscale(img)
        arr = np.array(img).astype(np.float32) / 255.0  # [0,1]
        arr = torch.from_numpy(arr)[None, None]  # [1,1,H,W]
        if processing_res > 0:
            arr = torch.nn.functional.interpolate(
                arr, size=processing_res, mode='bilinear')
        return arr, img.size
    else:
        img = img.convert('RGB')
        if processing_res > 0:
            img, orig_size = resize_max_res(img, processing_res)
        else:
            orig_size = img.size
        
        arr = np.array(img)
        arr = rearrange(arr, 'h w c -> c h w')
        arr = arr / 127.5 - 1.0  # Normalize to [-1,1]
        return torch.from_numpy(arr).float()[None], orig_size  # [1,C,H,W]

def calculate_metrics(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """Calculate comprehensive depth estimation metrics"""
    if mask is None:
        mask = gt_depth > 0  # Only evaluate valid pixels
    
    pred = pred_depth * mask
    gt = gt_depth * mask
    eps = 1e-6  # Small value to avoid division by zero
    
    # Standard metrics
    rmse = torch.sqrt(torch.mean((pred - gt) ** 2))
    rel = torch.mean(torch.abs(pred - gt) / (gt + eps))
    
    # Threshold metrics
    threshold = torch.max((pred+eps)/(gt+eps), (gt+eps)/(pred+eps))
    delta1 = (threshold < 1.25).float().mean() * 100
    delta2 = (threshold < 1.25**2).float().mean() * 100
    delta3 = (threshold < 1.25**3).float().mean() * 100
    
    # Absolute accuracy
    abs_diff = torch.abs(pred - gt)
    acc5cm = (abs_diff < 0.05).float().mean() * 100
    acc10cm = (abs_diff < 0.10).float().mean() * 100
    
    # RBSREL
    log_diff = torch.log(pred + eps) - torch.log(gt + eps)
    rbsrel = torch.mean(torch.abs(log_diff)) * 100
    
    return {
        'RMSE': rmse.item(),
        'REL': rel.item(),
        'δ1': delta1.item(),
        'δ2': delta2.item(),
        'δ3': delta3.item(),
        'Accuracy@5cm': acc5cm.item(),
        'Accuracy@10cm': acc10cm.item(),
        'RBSREL': rbsrel.item()
    }

def save_depth_map(
    depth: np.ndarray,
    output_path: str,
    orig_size: Optional[Tuple[int, int]] = None,
    colormap: str = 'magma'
) -> None:
    """Save depth map with optional colormap and resizing"""
    if colormap:
        depth = plt.get_cmap(colormap)(depth, bytes=True)[..., :3]
    else:
        depth = (depth * 255).astype(np.uint8)
    
    img = Image.fromarray(depth)
    if orig_size and img.size != orig_size:
        img = img.resize(orig_size, Image.BILINEAR)
    img.save(output_path)
    print(f"Saved depth map to {output_path}")

class DepthDataset(Dataset):
    """Dataset class for paired RGB-depth evaluation"""
    def __init__(self, image_dir: str, depth_dir: str, processing_res: int = -1):
        self.image_paths = sorted([
            os.path.join(image_dir, f) 
            for f in os.listdir(image_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        self.depth_paths = sorted([
            os.path.join(depth_dir, f) 
            for f in os.listdir(depth_dir) 
            if f.lower().endswith('.png')
        ])
        self.processing_res = processing_res
        assert len(self.image_paths) == len(self.depth_paths), \
            f"Mismatched counts: {len(self.image_paths)} images vs {len(self.depth_paths)} depth maps"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image, _ = load_image(self.image_paths[idx], self.processing_res, is_depth=False)
        depth, _ = load_image(self.depth_paths[idx], self.processing_res, is_depth=True)
        return {
            'image': image.squeeze(0),  # Remove batch dim for DataLoader
            'depth': depth.squeeze(0),
            'image_path': self.image_paths[idx]
        }

def evaluate_on_dataset(
    model: DepthFM,
    dataloader: DataLoader,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> Dict[str, float]:
    """Evaluate model on entire dataset"""
    metrics = {
        'RMSE': 0, 'REL': 0, 'δ1': 0, 'δ2': 0, 'δ3': 0,
        'Accuracy@5cm': 0, 'Accuracy@10cm': 0, 'RBSREL': 0
    }
    count = 0
    
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            gt_depths = batch['depth'].to(device)
            
            # Generate predictions
            pred_depths = model.predict_depth(images)
            
            # Compute metrics
            batch_metrics = calculate_metrics(pred_depths, gt_depths)
            
            # Accumulate
            for k in metrics:
                metrics[k] += batch_metrics[k]
            count += 1
    
    # Average metrics
    for k in metrics:
        metrics[k] /= count
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="DepthFM Inference")
    # Model parameters
    parser.add_argument("--ckpt", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16", help="Compute precision")
    
    # Evaluation mode selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--img", type=str, help="Path to single input image")
    group.add_argument("--eval_dataset", action="store_true", help="Enable dataset evaluation mode")
    
    # Single image parameters
    parser.add_argument("--gt_depth", type=str, default="", help="Path to ground truth depth map")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save results")
    
    # Dataset parameters
    parser.add_argument("--image_dir", type=str, help="Directory containing input images")
    parser.add_argument("--depth_dir", type=str, help="Directory containing ground truth depth maps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    
    # Common parameters
    parser.add_argument("--num_steps", type=int, default=4, help="Number of ODE solver steps")
    parser.add_argument("--ensemble_size", type=int, default=4, help="Number of ensemble members")
    parser.add_argument("--processing_res", type=int, default=512, help="Processing resolution (longer edge)")
    parser.add_argument("--no_color", action="store_true", help="Output grayscale depth map")
    
    args = parser.parse_args()

    # Initialize model
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dtype = get_dtype_from_str(args.dtype)
    
    print(f"\nInitializing DepthFM on {device} with {args.dtype} precision")
    model = DepthFM(args.ckpt).to(device).eval()
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    if args.eval_dataset:
        # Dataset evaluation mode
        print(f"\nEvaluating on dataset:")
        print(f"  Images: {args.image_dir}")
        print(f"  Depth maps: {args.depth_dir}")
        print(f"  Batch size: {args.batch_size}")
        
        dataset = DepthDataset(args.image_dir, args.depth_dir, args.processing_res)
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        metrics = evaluate_on_dataset(model, dataloader, device, dtype)
        
        print("\n=== Dataset Evaluation Results ===")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        
        # Save metrics to file
        metrics_file = os.path.join(args.output_dir, "dataset_metrics.txt")
        with open(metrics_file, 'w') as f:
            for name, value in metrics.items():
                f.write(f"{name}: {value:.4f}\n")
        print(f"\nSaved metrics to {metrics_file}")
    else:
        # Single image mode
        print(f"\nProcessing single image: {args.img}")
        try:
            image, orig_size = load_image(args.img, args.processing_res)
            image = image.to(device)
        except Exception as e:
            print(f"Error loading image: {e}")
            return

        # Load ground truth if available
        gt_depth = None
        if args.gt_depth:
            try:
                gt_depth, _ = load_image(args.gt_depth, args.processing_res, is_depth=True)
                gt_depth = gt_depth.to(device)
            except Exception as e:
                print(f"Warning: Could not load GT depth - {e}")

        # Run inference
        try:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
                depth = model.predict_depth(
                    image, 
                    num_steps=args.num_steps, 
                    ensemble_size=args.ensemble_size
                )
                
                if gt_depth is not None:
                    metrics = calculate_metrics(depth, gt_depth)
                    print("\n=== Depth Estimation Metrics ===")
                    for name, value in metrics.items():
                        print(f"{name}: {value:.4f}")
                    
                    # Save metrics to file
                    base_name = os.path.splitext(os.path.basename(args.img))[0]
                    metrics_file = os.path.join(args.output_dir, f"{base_name}_metrics.txt")
                    with open(metrics_file, 'w') as f:
                        for name, value in metrics.items():
                            f.write(f"{name}: {value:.4f}\n")
                    print(f"Saved metrics to {metrics_file}")

            # Save output
            base_name = os.path.splitext(os.path.basename(args.img))[0]
            output_path = os.path.join(args.output_dir, f"{base_name}_depth.png")
            save_depth_map(
                depth.squeeze().cpu().numpy(),
                output_path,
                orig_size,
                colormap=None if args.no_color else 'magma'
            )

        except RuntimeError as e:
            print(f"\nError during inference: {e}")
            if "CUDA out of memory" in str(e):
                print("Try reducing:")
                print("- Image resolution (--processing_res)")
                print("- Ensemble size (--ensemble_size)")
            return

if __name__ == "__main__":
    main()
