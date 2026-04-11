import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


INPUT_IMAGE = "text.bmp"
OUTPUT_DIR = "symbols"
PROFILE_DIR = "profiles"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROFILE_DIR, exist_ok=True)

def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY_INV)
    return img, binary

def get_profiles(img):
    profile_x = img.sum(axis=0)
    profile_y = img.sum(axis=1)
    return profile_x, profile_y

def save_profile(profile, name, axis):
    plt.figure()

    if axis == "X":
        plt.bar(range(len(profile)), profile)
        plt.xlabel("X")
        plt.ylabel("Сумма черных пикселей")
    else:
        plt.barh(range(len(profile)), profile)
        plt.gca().invert_yaxis()
        plt.ylabel("Y")
        plt.xlabel("Сумма черных пикселей")

    plt.title(f"{name} profile {axis}")
    plt.savefig(f"{PROFILE_DIR}/{name}_{axis}.png")
    plt.close()


#сегментация строк
def segment_lines(binary):
    profile_y = binary.sum(axis=1)

    lines = []
    in_line = False

    for i, val in enumerate(profile_y):
        if val > 0 and not in_line:
            start = i
            in_line = True

        elif val == 0 and in_line:
            end = i
            lines.append((start, end))
            in_line = False

    if in_line:
        lines.append((start, len(profile_y)))

    return lines


#сегментация символов
def segment_symbols(binary):
    profile_x = binary.sum(axis=0)

    segments = []
    in_symbol = False

    for i, val in enumerate(profile_x):
        if val > 0 and not in_symbol:
            start = i
            in_symbol = True

        elif val == 0 and in_symbol:
            end = i
            segments.append((start, end))
            in_symbol = False

    if in_symbol:
        segments.append((start, len(profile_x)))

    return segments


def main():
    original, binary = preprocess(INPUT_IMAGE)

    px, py = get_profiles(binary)
    save_profile(px, "text", "X")
    save_profile(py, "text", "Y")

    results = []

    lines = segment_lines(binary)

    for line_idx, (y1, y2) in enumerate(lines):
        line_img = binary[y1:y2, :]

        segments = segment_symbols(line_img)

        for i, (x1, x2) in enumerate(segments):
            symbol = line_img[:, x1:x2]

            coords = np.column_stack(np.where(symbol > 0))
            if coords.size == 0:
                continue

            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            symbol = symbol[y_min:y_max+1, x_min:x_max+1]

            name = f"line{line_idx+1}_symbol{i+1}"

            cv2.imwrite(f"{OUTPUT_DIR}/{name}.png", symbol * 255)

            px_s, py_s = get_profiles(symbol)
            save_profile(px_s, name, "X")
            save_profile(py_s, name, "Y")

            results.append({"name": name})

    print("Найдено символов:", len(results))

if __name__ == "__main__":
    main()