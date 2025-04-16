# DepthFM

## Revisiting DepthFM: A High-Fidelity and Efficient Flow Matching Framework for Monocular Depth Estimation

🔗 [Sample Depth Estimation](https://github.com/Tipusultan199/DepthFM/blob/main/2.png)

### 🔑 File: `depthfm-v1.ckpt`

📎 Download here: [depthfm-v1.ckpt (OneDrive link)](https://sluedu-my.sharepoint.com/:f:/g/personal/tipu_sultan_slu_edu/EndixeYj6dRPuhKia0kdI1sB5z_EdxYzd-C5YiO8VEWT8Q?e=TaUA2E)

---

**Improved DepthFM++** is an enhanced version of the original [DepthFM](https://arxiv.org/abs/2403.13788), designed for high-fidelity, fast, and flexible monocular depth estimation. This version incorporates robust flow-matching refinements to boost accuracy and reliability, especially in high-resolution or real-time scenarios.

---

## 🔍 Key Highlights

- Optimized ODE solver for smoother trajectory integration  
- Improved ensemble sampling for robust uncertainty estimation  
- Adaptive resolution and percentile-based normalization  
- Significantly better depth metrics at close ranges  

---

## 🚀 Quick Start

### Prerequisites

- NVIDIA GPU (16GB+ VRAM recommended)  
- Python 3.8+  
- CUDA 11.7+  

---

### ⚙️ Installation

```bash
# Unzip and enter the directory
unzip DepthFM-Improved.zip
cd Improved_code(DepthFM)

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows

# Install required packages
pip install -r requirements.txt

### File Structure

DepthFM/
├── Images/                        # Input images (e.g., dog.png)
├── Improved_code(DepthFM)/       # Enhanced DepthFM code
│   ├── checkpoints/              # Model weights
│   ├── inference.py              # Run inference on images
│   └── ...
├── 1.png                         # Depth map output (sample)
├── 2.png                         # Model prediction visualization
└── README.md                     # Project documentation


### Running Inference
## Single Image
python inference.py \
  --ckpt checkpoints/depthfm-v1.ckpt \
  --img Images/dog.png \
  --output_dir outputs \
  --num_steps 8 \
  --ensemble_size 5

## Batch Processing
python inference.py \
  --eval_dataset \
  --image_dir Images/rgb/ \
  --depth_dir Images/depth/ \
  --batch_size 4


