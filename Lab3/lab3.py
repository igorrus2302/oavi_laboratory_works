import os
import requests
from PIL import Image
import numpy as np


origin = "https://www.slavcorpora.ru"
sample_id = "b008ae91-32cf-4d7d-84e4-996144e4edb7"

DOWNLOAD_LIMIT = 3

INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

sample = requests.get(f"{origin}/api/samples/{sample_id}").json()
image_paths = [f"{origin}/images/{p['filename']}" for p in sample["pages"]]

image_paths = image_paths[:DOWNLOAD_LIMIT]

saved_files = []

for i, url in enumerate(image_paths):
    response = requests.get(url)

    jpg_path = os.path.join(INPUT_DIR, f"img_{i}.jpg")
    png_path = os.path.join(INPUT_DIR, f"img_{i}.png")

    with open(jpg_path, "wb") as f:
        f.write(response.content)

    img = Image.open(jpg_path).convert("L")
    img.save(png_path, "PNG")

    os.remove(jpg_path)

    saved_files.append(png_path)

print("Скачивание и перевод завершены")

def median_cross_filter(img_array):
    h, w = img_array.shape
    result = np.zeros_like(img_array)

    binary = (img_array > 127).astype(np.uint8)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            neighbors = [
                binary[i, j],
                binary[i-1, j],
                binary[i+1, j],
                binary[i, j-1],
                binary[i, j+1]
            ]

            result[i, j] = 1 if sum(neighbors) >= 3 else 0

    return result * 255

print("Фильтрация")

for path in saved_files:
    filename = os.path.basename(path).replace(".png", "")

    img = Image.open(path).convert("L")
    arr = np.array(img)

    filtered = median_cross_filter(arr)

    binary_original = (arr > 127).astype(np.uint8)
    binary_filtered = (filtered > 127).astype(np.uint8)

    diff = np.bitwise_xor(binary_original, binary_filtered) * 255

    Image.fromarray(filtered.astype(np.uint8)).save(
        os.path.join(OUTPUT_DIR, f"{filename}_filtered.png")
    )

    Image.fromarray(diff.astype(np.uint8)).save(
        os.path.join(OUTPUT_DIR, f"{filename}_diff.png")
    )

print("Успешно")