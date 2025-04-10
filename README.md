# DepthFM - Fast Monocular Depth Estimation with Flow Matching

This repository contains a fully working implementation of **DepthFM**, a fast and high-quality monocular depth estimation model based on flow matching and Stable Diffusion v2.1 priors. The setup is tested on GPU with CUDA and includes complete guidance for inference using your own images.

> ✅ Runs on GPU with PyTorch 2.1.0 + CUDA 12.1  
> ✅ Compatible with `xformers` 0.0.22.post7  
> ✅ Single-step inference, no diffusion sampling  
> ✅ Includes pretrained checkpoint and sample input

---

## 🧠 Paper Reference

**DepthFM: Fast Monocular Depth Estimation with Flow Matching**  

---

## 🧪 Example Output

| Input Image | Predicted Depth |
|-------------|-----------------|
| ![](assets/dog.png) | ![](assets/dog.png_depth.png) |

---

## 🖥️ System Requirements

- **OS**: Ubuntu 20.04 / 22.04
- **Python**: 3.10+
- **CUDA**: 12.1
- **GPU**: NVIDIA GPU with at least 8GB VRAM
- **Torch**: 2.1.0 + cu121
- **xFormers**: 0.0.22.post7

> 💡 This model **requires GPU**. CPU-only execution is not supported due to `xformers` FlashAttention ops.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Tipusultan199/DepthFM.git
cd DepthFM
