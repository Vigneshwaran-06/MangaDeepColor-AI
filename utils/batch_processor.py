import os
from utils.preprocess import preprocess_image
from PIL import Image
import numpy as np

def batch_colorize(input_folder, output_folder, model):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            pre_img = preprocess_image(img_path)
            input_array = np.array(pre_img).astype(np.float32) / 255.0
            input_array = input_array[..., np.newaxis]
            input_array = np.expand_dims(input_array, axis=0)
            colorized = model.predict(input_array)[0]
            colorized_img = (colorized * 255).astype(np.uint8)
            colorized_pil = Image.fromarray(colorized_img)
            colorized_pil.save(os.path.join(output_folder, filename))
