# Revisiting DepthFM: A High-Fidelity and Efficient Flow Matching Framework for Monocular Depth Estimation
![Sample Depth Estimation](assets/sample_comparison.jpg)

**DepthFM++** is an enhanced version of the original [DepthFM](https://arxiv.org/abs/2403.13788), designed for high-fidelity, fast, and flexible monocular depth estimation. This version incorporates robust flow-matching refinements to boost accuracy and reliability, especially in high-resolution or real-time scenarios.

---

## Key Highlights
- 🔁 Optimized ODE solver for smoother trajectory integration
- 🎲 Improved ensemble sampling for robust uncertainty estimation
- 📐 Adaptive resolution and percentile-based normalization
- ⚡ Achieves significantly better depth metrics at close ranges

---

##  Quick Start

###  Prerequisites
- NVIDIA GPU (16GB+ VRAM recommended)
- Python 3.8+
- CUDA 11.7+

###  Installation
```bash
unzip DepthFM-Improved.zip
cd DepthFM-Improved

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows

# Install dependencies
pip install -r requirements.txt

### File Structure

DepthFM-Improved/
├── assets/               # Sample RGB/depth images
├── checkpoints/          # Pretrained model weights
│   └── depthfm-v1.5.ckpt
├── depthfm/              # Core flow matching code
├── outputs/              # Saved inference results
└── inference.py          # Inference script


### Running Inference
## Single Image

python inference.py \
  --ckpt checkpoints/depthfm-v1.5.ckpt \
  --img assets/dog.png \
  --output_dir outputs \
  --num_steps 8 \
  --ensemble_size 5


###  Batch Processing

python inference.py \
  --eval_dataset \
  --image_dir assets/rgb_images \
  --depth_dir assets/depth_maps \
  --batch_size 4



**Next Steps:**
- Make sure the `assets/sample_comparison.jpg` image exists to show in the preview.
- Include the `requirements.txt` and `LICENSE` files in your repo root.

Let me know if you want a version with a GitHub Actions badge, Colab notebook, or citation block for academic use.

