from PIL import Image
import os

def resize_images(input_folder, output_folder, size=(128, 128)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path).convert("RGB")

            width, height = img.size
            short_side = min(width, height)

            left = (width - short_side) // 2
            top = (height - short_side) // 2
            right = left + short_side
            bottom = top + short_side

            img = img.crop((left, top, right, bottom))
            img = img.resize(size, Image.LANCZOS)
            img.save(os.path.join(output_folder, filename))

resize_images('content/raw', 'content/processed')
resize_images('style/raw', 'style/processed')