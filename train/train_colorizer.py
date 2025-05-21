import os
import numpy as np
from PIL import Image
from models.colorizer import unet_colorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

def load_data(data_dir):
    X, Y = [], []
    for file in os.listdir(data_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                
                gray = Image.open(os.path.join(data_dir, file)).convert("L").resize((256, 256))
                color = Image.open(os.path.join(data_dir, file)).convert("RGB").resize((256, 256))

                gray_arr = np.expand_dims(np.array(gray) / 255.0, axis=-1)
                color_arr = np.array(color) / 255.0

                X.append(gray_arr)
                Y.append(color_arr)
            except Exception as e:
                print(f"[ERROR] Skipping {file}: {e}")
    
    print(f"[INFO] Loaded {len(X)} image(s) from {data_dir}")
    return np.array(X), np.array(Y)

def train_model(data_dir, epochs=20):
    X, Y = load_data(data_dir)

    if len(X) == 0 or len(Y) == 0:
        raise ValueError(f"No images found in {data_dir}. Please check the path or image files.")

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

    model = unet_colorizer()
    model.compile(optimizer=Adam(1e-4), loss="mse", metrics=["mae"])

    print("[INFO] Starting training...")
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=8)
    
    model.save("models/colorizer_trained.h5")
    print("[INFO] Model saved to models/colorizer_trained.h5")

if __name__ == "__main__":
    train_model("data/manga_train/")
