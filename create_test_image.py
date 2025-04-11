# create_test_image.py
import numpy as np
from PIL import Image
import argparse

def create_test_image(output_path):
    # Create a simple RGB image with colored squares
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    img[100:200, 100:200] = [255, 0, 0]    # Red square
    img[300:400, 200:300] = [0, 255, 0]    # Green square
    img[200:300, 300:400] = [0, 0, 255]    # Blue square
    Image.fromarray(img).save(output_path)
    print(f"Created test image at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output PNG path")
    args = parser.parse_args()
    create_test_image(args.output)
