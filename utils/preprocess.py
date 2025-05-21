import cv2
from PIL import Image
import numpy as np

def preprocess_image(uploaded_file):
    # Read image as bytes from uploaded file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    # Decode image from bytes
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Denoise, resize, normalize
    denoised = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    resized = cv2.resize(denoised, (256, 256))
    norm = resized.astype(np.float32) / 255.0

    # Convert back to PIL for display or model
    pil_img = Image.fromarray((norm * 255).astype(np.uint8))
    return pil_img
