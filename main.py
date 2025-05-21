
---

## 5. main.py (entry point)

```python
import argparse
from models.colorizer import unet_colorizer
from utils.batch_processor import batch_colorize

def main():
    parser = argparse.ArgumentParser(description="Manga Magic Batch Colorizer")
    parser.add_argument("--input", type=str, required=True, help="Input folder with grayscale manga images")
    parser.add_argument("--output", type=str, required=True, help="Output folder for colorized images")
    args = parser.parse_args()

    model = unet_colorizer()
    model.load_weights('models/colorizer_weights.h5')  # Adjust path if needed

    batch_colorize(args.input, args.output, model)

if __name__ == "__main__":
    main()
