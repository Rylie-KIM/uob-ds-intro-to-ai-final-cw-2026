import subprocess
import os

class EPSConverter:
    """
    A class to convert EPS files to PNG format using ImageMagick.
    """

    def __init__(self, imagemagick_path=None):
        
        self.imagemagick_path = imagemagick_path or "magick"

    def convert_to_png(self, eps_path: str, png_path: str) -> None:
        if not os.path.exists(eps_path):
            raise FileNotFoundError(f"The EPS file '{eps_path}' was not found.")

        try:
            cmd = [self.imagemagick_path, eps_path, png_path]
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise subprocess.CalledProcessError(e.returncode, e.cmd, e.output, e.stderr)
        except FileNotFoundError:
            raise Exception("ImageMagick (magick.exe) not found. Please ensure ImageMagick is installed and in PATH, or provide the path to magick.exe.")

if __name__ == "__main__":
    eps_dir = r"C:\Users\user\uob-ds-intro-to-ai-final-cw-2026\src\data_generation\type-a\type-a-dataset"
    png_dir = r"C:\Users\user\uob-ds-intro-to-ai-final-cw-2026\src\data_generation\type-a\type-a-dataset-png"

    os.makedirs(png_dir, exist_ok=True)

    imagemagick_path = r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"
    converter = EPSConverter(imagemagick_path)

    eps_files = [f for f in os.listdir(eps_dir) if f.endswith('.eps')]

    batch_size = 500
    for i in range(0, len(eps_files), batch_size):
        batch = eps_files[i:i + batch_size]
        for eps_file in batch:
            eps_path = os.path.join(eps_dir, eps_file)
            png_path = os.path.join(png_dir, os.path.splitext(eps_file)[0] + '.png')
            try:
                converter.convert_to_png(eps_path, png_path)
                print(f"Successfully converted {eps_file} to PNG.")
            except Exception as e:
                print(f"Failed to convert {eps_file}: {e}")