import os
import requests
from PIL import Image
import numpy as np
import random


origin = "https://www.slavcorpora.ru"
sample_id = "b008ae91-32cf-4d7d-84e4-996144e4edb7"

DOWNLOAD_LIMIT = 3

INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def choose_mode():
    print("Выберите режим загрузки изображений:")
    print("1 — Последовательная загрузка")
    print("2 — Случайная загрузка")

    while True:
        choice = input("Введите 1 или 2: ").strip()

        if choice == "1":
            return "sequential"
        elif choice == "2":
            return "random"
        else:
            print("Ошибка! Введите 1 или 2.")

def download_images(mode):
    sample = requests.get(f"{origin}/api/samples/{sample_id}").json()
    pages = sample["pages"]

    if mode == "sequential":
        selected_pages = pages[:DOWNLOAD_LIMIT]

    elif mode == "random":
        if DOWNLOAD_LIMIT > len(pages):
            raise ValueError("DOWNLOAD_LIMIT больше количества доступных изображений")
        selected_pages = random.sample(pages, DOWNLOAD_LIMIT)

    else:
        raise ValueError("Неверный режим")

    saved_files = []

    for i, p in enumerate(selected_pages):
        url = f"{origin}/images/{p['filename']}"

        jpg_path = os.path.join(INPUT_DIR, f"img_{i}.jpg")
        png_path = os.path.join(INPUT_DIR, f"img_{i}.png")

        response = requests.get(url)

        with open(jpg_path, "wb") as f:
            f.write(response.content)

        img = Image.open(jpg_path).convert("L")
        img.save(png_path, "PNG")

        os.remove(jpg_path)

        saved_files.append(png_path)

    return saved_files

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

def main():
    mode = choose_mode()

    saved_files = download_images(mode)

    print("Скачивание и перевод завершены")
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


if __name__ == "__main__":
    main()