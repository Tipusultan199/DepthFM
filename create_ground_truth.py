# File: create_ground_truth.py
import numpy as np
from PIL import Image
import argparse

def create_synthetic_depth(output_path: str, width: int = 512, height: int = 512):
    """Generate horizontal depth gradient"""
    depth = np.linspace(0, 255, width, dtype=np.uint8)
    depth = np.tile(depth, (height, 1))
    Image.fromarray(depth).save(output_path)
    print(f"Saved synthetic depth map to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="assets/dog_depth_gt.png")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()
    
    create_synthetic_depth(args.output, args.width, args.height)
