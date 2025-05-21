# Manga Magic

Python AI tool to colorize manga images with advanced deep learning super resolution and denoising.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


# MangaMagic 🎨
A U-Net based deep learning application for automatic manga colorization.

## 🧠 Model Architecture
- Backbone: U-Net
- Loss: MAE + SSIM Loss
- Optimizer: Adam
- Input: Grayscale Manga (256x256)
- Output: RGB Color Manga

## 🧪 Training
- Dataset: Custom Manga dataset
- Epochs: 20
- Batch Size: 16
- Metrics: PSNR, SSIM
