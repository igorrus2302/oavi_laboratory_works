import numpy as np
from PIL import Image
import os
import requests
import random

ORIGIN = "https://www.slavcorpora.ru"
SAMPLE_ID = "b008ae91-32cf-4d7d-84e4-996144e4edb7"
DOWNLOAD_LIMIT = 3
INPUT_FOLDER = 'input_images'
OUTPUT_FOLDER = 'output_images'

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
    os.makedirs(INPUT_FOLDER, exist_ok=True)

    sample = requests.get(f"{ORIGIN}/api/samples/{SAMPLE_ID}").json()
    pages = sample["pages"]

    if mode == "sequential":
        selected_pages = pages[:DOWNLOAD_LIMIT]

    elif mode == "random":
        if DOWNLOAD_LIMIT > len(pages):
            raise ValueError("DOWNLOAD_LIMIT больше количества изображений")
        selected_pages = random.sample(pages, DOWNLOAD_LIMIT)

    else:
        raise ValueError("Неверный режим")

    count = 0

    for idx, p in enumerate(selected_pages):
        url = f"{ORIGIN}/images/{p['filename']}"

        filename = f"img_{idx:04d}.png"
        filepath_png = os.path.join(INPUT_FOLDER, filename)

        if not os.path.exists(filepath_png):
            img_data = requests.get(url).content

            temp_path = os.path.join(INPUT_FOLDER, f"temp_{idx}.jpg")
            with open(temp_path, 'wb') as f:
                f.write(img_data)

            try:
                img = Image.open(temp_path).convert('RGB')
                img = img.resize((512, 512))
                img.save(filepath_png, 'PNG')
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        count += 1

    print(f"Downloaded and converted {count} images to PNG")

def rgb_to_grayscale(img_array):
    r = img_array[:, :, 0].astype(float)
    g = img_array[:, :, 1].astype(float)
    b = img_array[:, :, 2].astype(float)
    gray = 0.3 * r + 0.59 * g + 0.11 * b
    return gray.astype(np.uint8)

def otsu_threshold(window):
    hist = np.zeros(256)
    for val in window.flatten():
        hist[int(val)] += 1

    total = window.size
    sum_total = sum(i * hist[i] for i in range(256))

    sumB = 0
    wB = 0
    max_var = 0
    threshold = 0

    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break

        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF

        var_between = wB * wF * (mB - mF) ** 2

        if var_between > max_var:
            max_var = var_between
            threshold = t

    return threshold

def eikvil_binarization(gray_img, small_size=3, large_size=15, eps=15):
    h, w = gray_img.shape
    result = np.zeros((h, w), dtype=np.uint8)

    r_small = small_size // 2
    r_large = large_size // 2

    for i in range(0, h, small_size):
        for j in range(0, w, small_size):
            x1 = max(i - r_large, 0)
            x2 = min(i + r_large + 1, h)
            y1 = max(j - r_large, 0)
            y2 = min(j + r_large + 1, w)

            large_window = gray_img[x1:x2, y1:y2]
            T = otsu_threshold(large_window)

            class1 = large_window[large_window <= T]
            class2 = large_window[large_window > T]

            if len(class1) == 0 or len(class2) == 0:
                continue

            m1 = np.mean(class1)
            m2 = np.mean(class2)

            xs1 = i
            xs2 = min(i + small_size, h)
            ys1 = j
            ys2 = min(j + small_size, w)

            small_window = gray_img[xs1:xs2, ys1:ys2]

            if abs(m1 - m2) >= eps:
                for x in range(xs1, xs2):
                    for y in range(ys1, ys2):
                        result[x, y] = 255 if gray_img[x, y] > T else 0
            else:
                mean_val = m1 if abs(np.mean(small_window) - m1) < abs(np.mean(small_window) - m2) else m2
                for x in range(xs1, xs2):
                    for y in range(ys1, ys2):
                        result[x, y] = 255 if mean_val > T else 0

    return result

def process_image(path, out_prefix):
    img = Image.open(path).convert('RGB')
    img_np = np.array(img)

    gray = rgb_to_grayscale(img_np)
    Image.fromarray(gray).save(out_prefix + '_gray.bmp')

    binary = eikvil_binarization(gray)
    Image.fromarray(binary).save(out_prefix + '_binary.bmp')

    print(f"Processed: {path}")

if __name__ == '__main__':
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    mode = choose_mode()
    download_images(mode)

    for file in os.listdir(INPUT_FOLDER):
        if file.endswith('.png') or file.endswith('.bmp'):
            in_path = os.path.join(INPUT_FOLDER, file)
            out_prefix = os.path.join(OUTPUT_FOLDER, file.split('.')[0])
            process_image(in_path, out_prefix)