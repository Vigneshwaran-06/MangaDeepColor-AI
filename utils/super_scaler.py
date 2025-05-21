import cv2

def upscale_image(image, scale=2):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    try:
        sr.readModel("models/EDSR_x2.pb")  # Download EDSR model separately
        sr.setModel("edsr", scale)
        upscaled = sr.upsample(image)
        return upscaled
    except Exception as e:
        print(f"Super-resolution model failed: {e}")
        return image
