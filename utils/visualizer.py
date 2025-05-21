from PIL import Image

def get_comparison_image(original_img, colorized_img):
    # Ensure both images are same size
    original_img = original_img.resize((256, 256))
    colorized_img = colorized_img.resize((256, 256))

    # Create new image with double width for side-by-side
    comparison_img = Image.new('RGB', (512, 256))
    comparison_img.paste(original_img, (0, 0))
    comparison_img.paste(colorized_img, (256, 0))

    return comparison_img
