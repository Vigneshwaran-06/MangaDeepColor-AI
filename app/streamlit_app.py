import streamlit as st
from PIL import Image
import numpy as np
import os
import io
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

from models.colorizer import unet_colorizer  # add other models here when ready
from utils.preprocess import preprocess_image
from utils.visualizer import get_comparison_image

# === UI Setup ===
st.set_page_config(page_title="Manga Magic", layout="centered")
st.title("🪄 Manga DeepColor - AI Manga Colorizer")




# === Model Path Setup ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "colorizer_trained.h5")

# === Upload Image ===
uploaded_file = st.file_uploader("📂 Upload a grayscale manga image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    grayscale_img = Image.open(uploaded_file).convert('L')
    st.image(grayscale_img, caption="🖤 Grayscale Manga", use_container_width=True)

    # === Model Selection ===
    model_option = st.selectbox("🧠 Choose Colorization Model", ["U-Net Colorizer"])  # extend options later

    if st.button("🎨 Colorize"):
        with st.spinner("Colorizing... please wait..."):
            # === Load Model ===
            if model_option == "U-Net Colorizer":
                model = unet_colorizer()
                model.load_weights(MODEL_PATH)
            else:
                st.error("Model not found.")
                st.stop()

            # === Preprocess Image ===
            uploaded_file.seek(0)
            pre_img = preprocess_image(uploaded_file)
            input_array = np.array(pre_img).astype(np.float32) / 255.0
            input_array = input_array[..., np.newaxis]
            input_array = np.expand_dims(input_array, axis=0)

            # === Prediction ===
            colorized = model.predict(input_array)[0]
            colorized_img = (colorized * 255).astype(np.uint8)
            colorized_pil = Image.fromarray(colorized_img)

            # === Display Comparison ===
            comparison = get_comparison_image(grayscale_img.convert("RGB"), colorized_pil)
            st.image(comparison, caption="🎭 Before vs After", use_container_width=True)

            # === Metrics Calculation ===
            original_resized = np.array(pre_img.resize((256, 256))).astype(np.uint8)
            colorized_resized = np.array(colorized_pil.resize((256, 256))).astype(np.uint8)

            if len(colorized_resized.shape) == 3:
                colorized_gray = cv2.cvtColor(colorized_resized, cv2.COLOR_RGB2GRAY)
            else:
                colorized_gray = colorized_resized

            psnr_val = psnr(original_resized, colorized_gray, data_range=255)
            ssim_val = ssim(original_resized, colorized_gray, data_range=255, win_size=7)

            st.markdown("### 📊 Quality Metrics")
            st.write(f"**PSNR**: {psnr_val:.2f} dB")
            st.write(f"**SSIM**: {ssim_val:.4f}")

            # === Download Button ===
            img_bytes = io.BytesIO()
            colorized_pil.save(img_bytes, format="PNG")
            st.download_button("📥 Download Colorized Image", data=img_bytes.getvalue(),
                               file_name="colorized_manga.png", mime="image/png")

            st.success("✨ Colorization Complete!")

            # === Optional: Save History (commented for now) ===
            # history.append({
            #     "psnr": psnr_val,
            #     "ssim": ssim_val,
            #     "image": colorized_pil
            # })

