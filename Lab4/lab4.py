import os
import requests
import numpy as np
import cv2
from PIL import Image
import random


origin = "https://www.slavcorpora.ru"
sample_id = "b008ae91-32cf-4d7d-84e4-996144e4edb7"

INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"

MAX_IMAGES = 1
THRESHOLD = 100

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
            print("Ошибка! Введите только 1 или 2.")

def download_images(mode):
    sample = requests.get(f"{origin}/api/samples/{sample_id}").json()
    pages = sample["pages"]

    if mode == "sequential":
        selected_pages = pages[:MAX_IMAGES]

    elif mode == "random":
        if MAX_IMAGES > len(pages):
            raise ValueError("MAX_IMAGES больше, чем доступных изображений")
        selected_pages = random.sample(pages, MAX_IMAGES)

    else:
        raise ValueError("Неверный режим загрузки")

    image_paths = []

    for i, p in enumerate(selected_pages):
        url = f"{origin}/images/{p['filename']}"
        jpeg_path = os.path.join(INPUT_DIR, f"img_{i}.jpg")

        response = requests.get(url)
        with open(jpeg_path, "wb") as f:
            f.write(response.content)

        png_path = jpeg_path.replace(".jpg", ".png")
        Image.open(jpeg_path).convert("RGB").save(png_path)

        os.remove(jpeg_path)

        image_paths.append(png_path)

    return image_paths

Gx_kernel = np.array([
    [3, 10, 3],
    [0, 0, 0],
    [-3, -10, -3]
], dtype=np.float32)

Gy_kernel = np.array([
    [-3, 0, 3],
    [-10, 0, 10],
    [-3, 0, 3]
], dtype=np.float32)

def process_image(path, index):
    img_color = cv2.imread(path)
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    Gx = cv2.filter2D(gray, -1, Gx_kernel)
    Gy = cv2.filter2D(gray, -1, Gy_kernel)

    G = np.sqrt(Gx.astype(np.float32)**2 + Gy.astype(np.float32)**2)

    def normalize(img):
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    Gx_norm = normalize(Gx)
    Gy_norm = normalize(Gy)
    G_norm = normalize(G)

    _, G_bin = cv2.threshold(G_norm, THRESHOLD, 255, cv2.THRESH_BINARY)

    cv2.imwrite(f"{OUTPUT_DIR}/img_{index}_color.png", img_color)
    cv2.imwrite(f"{OUTPUT_DIR}/img_{index}_gray.png", gray)
    cv2.imwrite(f"{OUTPUT_DIR}/img_{index}_Gx.png", Gx_norm)
    cv2.imwrite(f"{OUTPUT_DIR}/img_{index}_Gy.png", Gy_norm)
    cv2.imwrite(f"{OUTPUT_DIR}/img_{index}_G.png", G_norm)
    cv2.imwrite(f"{OUTPUT_DIR}/img_{index}_binary.png", G_bin)

def main():
    mode = choose_mode()
    paths = download_images(mode)

    for i, path in enumerate(paths):
        process_image(path, i)

    print("Успешно!")


if __name__ == "__main__":
    main()