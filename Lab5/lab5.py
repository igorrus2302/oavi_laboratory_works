import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

FONT_PATH = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
FONT_SIZE = 52
OUTPUT_DIR = "symbols"
PROFILE_DIR = "profiles"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROFILE_DIR, exist_ok=True)

alphabet = list("абвгдежѕзиiклмнопрстуфхцчшщъыьѣюѵѯѱѡѳѧѫ")


def crop_binary(binary):
    coords = np.column_stack(np.where(binary > 0))
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    return binary[y_min:y_max+1, x_min:x_max+1]


def generate_symbols():
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    for char in alphabet:
        img = Image.new('L', (100, 100), color=255)
        draw = ImageDraw.Draw(img)

        w, h = draw.textbbox((0, 0), char, font=font)[2:]
        draw.text(((100 - w) // 2, (100 - h) // 2), char, font=font, fill=0)

        # бинаризация
        img_np = np.array(img)
        _, binary = cv2.threshold(img_np, 127, 1, cv2.THRESH_BINARY_INV)

        # обрезка
        cropped = crop_binary(binary)

        # обратно в 0-255
        cropped_img = (255 - cropped * 255).astype(np.uint8)

        cv2.imwrite(f"{OUTPUT_DIR}/{char}.png", cropped_img)


def preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY_INV)
    return binary


def extract_features(img):
    h, w = img.shape

    q1 = img[:h//2, :w//2].sum()
    q2 = img[:h//2, w//2:].sum()
    q3 = img[h//2:, :w//2].sum()
    q4 = img[h//2:, w//2:].sum()

    area = (h//2)*(w//2)
    q_rel = [q1/area, q2/area, q3/area, q4/area]

    y, x = np.indices(img.shape)
    weight = img.sum()

    cx = (x * img).sum() / weight
    cy = (y * img).sum() / weight

    cx_rel = cx / w
    cy_rel = cy / h

    Ix = ((y - cy)**2 * img).sum()
    Iy = ((x - cx)**2 * img).sum()

    Ix_rel = Ix / (h * w)
    Iy_rel = Iy / (h * w)

    profile_x = img.sum(axis=0)
    profile_y = img.sum(axis=1)

    return {
        "Верхняя левая четверть": q1,
        "Верхняя правая четверть": q2,
        "Нижняя левая четверть": q3,
        "Нижняя правая четверть": q4,

        "Удельный вес верхней левой четверти": q_rel[0],
        "Удельный вес верхней правой четверти": q_rel[1],
        "Удельный вес нижней левой четверти": q_rel[2],
        "Удельный вес нижней правой четверти": q_rel[3],

        "Центр тяжести x": cx,
        "Центр тяжести y": cy,

        "Нормированный центр тяжести x": cx_rel,
        "Нормированный центр тяжести y": cy_rel,

        "Момент инерции x": Ix,
        "Момент инерции y": Iy,

        "Нормированный момент инерции x": Ix_rel,
        "Нормированный момент инерции y": Iy_rel,

        "Профиль x": profile_x,
        "Профиль y": profile_y
    }


def save_profile(profile, name, axis):
    plt.figure()

    if axis == "X":
        plt.bar(range(len(profile)), profile)
        plt.xlabel("X")
        plt.ylabel("Сумма пикселей")

    else:
        plt.barh(range(len(profile)), profile)
        plt.ylabel("Y")
        plt.xlabel("Сумма пикселей")

    plt.title(f"{name} profile {axis}")

    plt.savefig(f"{PROFILE_DIR}/{name}_{axis}.png")
    plt.close()


def main():
    generate_symbols()

    rows = []

    for char in alphabet:
        path = f"{OUTPUT_DIR}/{char}.png"
        img = preprocess(path)

        features = extract_features(img)

        save_profile(features["Профиль x"], char, "X")
        save_profile(features["Профиль y"], char, "Y")

        row = {
            "Символ": char,
            **{k: v for k, v in features.items() if "Профиль" not in k}
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("features.csv", sep=";", index=False)

if __name__ == "__main__":
    main()