import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_metrics(original, colorized):
    original = np.array(original).astype("float32") / 255.0
    colorized = np.array(colorized).astype("float32") / 255.0
    return {
        "SSIM": ssim(original, colorized, channel_axis=2),
        "PSNR": psnr(original, colorized)
    }
