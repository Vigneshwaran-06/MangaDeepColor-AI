import os
from PIL import Image
from utils.metrics import calculate_metrics

def generate_report(input_folder, output_folder):
    report = []
    for file in os.listdir(input_folder):
        in_path = os.path.join(input_folder, file)
        out_path = os.path.join(output_folder, file)
        if os.path.exists(out_path):
            original = Image.open(in_path).convert("RGB")
            colorized = Image.open(out_path).convert("RGB")
            metrics = calculate_metrics(original, colorized)
            report.append((file, metrics["SSIM"], metrics["PSNR"]))
    return report
