## Generate gifs from the images in the images_* folders 
import os
from PIL import Image
import glob
import re

def make_gif_from_images(image_folder, output_gif, duration=100):
    # Get all image files in the folder
    image_files = sorted(
        glob.glob(os.path.join(image_folder, "step_*.png")),
        key=lambda x: int(re.search(r'step_(\d+)\.png', x).group(1))
    )

    # Load images
    images = [Image.open(img_file) for img_file in image_files]

    # Save as GIF
    if images:
        images[0].save(
            output_gif,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        print(f"Saved GIF: {output_gif}")
    else:
        print(f"No images found in {image_folder} to create GIF.")


if __name__ == "__main__":
    # Specify the folders to create GIFs from
    folders = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('images_')]

    for folder in folders:
        output_gif = f"gifs/{folder}.gif"
        make_gif_from_images(folder, output_gif, duration=100)