import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FONT_PATH    = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
FONT_SIZE    = 52
FONT_SIZE_EXP = 44

ALPHABET = list("абвгдежѕзиiклмнопрстуфхцчшщъыьѣюѵѯѱѡѳѧѫ")
RECOGNIZED_TEXT = "сегментация"

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def crop_binary(binary):
    coords = np.column_stack(np.where(binary > 0))
    if coords.size == 0:
        return binary
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return binary[y0:y1 + 1, x0:x1 + 1]


def extract_features(binary):
    h, w = binary.shape
    total = float(binary.sum())
    if total == 0:
        return np.zeros(5)
    y_idx, x_idx = np.indices(binary.shape)
    cx = (x_idx * binary).sum() / total / w
    cy = (y_idx * binary).sum() / total / h
    Ix = ((y_idx - cy * h) ** 2 * binary).sum() / (h ** 2 * total)
    Iy = ((x_idx - cx * w) ** 2 * binary).sum() / (w ** 2 * total)
    mass = total / (h * w)
    return np.array([mass, cx, cy, Ix, Iy], dtype=float)


def similarity(v1, v2):
    return 1.0 / (1.0 + np.linalg.norm(v1 - v2))


def render_char(char, font):
    canvas = Image.new('L', (300, 300), 255)
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 20), char, font=font, fill=0)
    arr = np.array(canvas)
    _, binary = cv2.threshold(arr, 127, 1, cv2.THRESH_BINARY_INV)
    return crop_binary(binary)


def generate_text_image(text, font_size):
    font = ImageFont.truetype(FONT_PATH, font_size)
    canvas = Image.new('L', (3000, 300), 255)
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 20), text, font=font, fill=0)
    arr = np.array(canvas)
    _, binary = cv2.threshold(arr, 127, 1, cv2.THRESH_BINARY_INV)
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = arr[rmin:rmax + 1, cmin:cmax + 1]
    out_path = os.path.join(OUTPUT_DIR, f"text_{font_size}pt.bmp")
    cv2.imwrite(out_path, cropped)
    return cropped, out_path


def segment_symbols(binary):
    profile_x = binary.sum(axis=0)
    segs, in_sym = [], False
    for i, val in enumerate(profile_x):
        if val > 0 and not in_sym:
            start = i; in_sym = True
        elif val == 0 and in_sym:
            segs.append((start, i)); in_sym = False
    if in_sym:
        segs.append((start, len(profile_x)))
    return segs


def extract_symbols_from_image(gray):
    _, binary = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY_INV)
    segs = segment_symbols(binary)
    symbols = []
    for (x1, x2) in segs:
        sym = binary[:, x1:x2]
        sym = crop_binary(sym)
        if sym.size > 0:
            symbols.append(sym)
    return symbols


def build_alphabet_db(font_size):
    font = ImageFont.truetype(FONT_PATH, font_size)
    db = {}
    for char in ALPHABET:
        binary = render_char(char, font)
        db[char] = extract_features(binary)
    return db


def classify(symbols, db):
    results = []
    for sym in symbols:
        fv = extract_features(sym)
        hyps = [(char, round(similarity(fv, ref), 4)) for char, ref in db.items()]
        hyps.sort(key=lambda x: x[1], reverse=True)
        results.append(hyps)
    return results


def print_and_save_results(results, true_text, file_path, label=""):
    recognized = "".join(hyps[0][0] for hyps in results)
    errors = sum(1 for a, b in zip(recognized, true_text) if a != b)
    errors += abs(len(recognized) - len(true_text))
    correct = len(true_text) - errors
    accuracy = correct / len(true_text) * 100 if true_text else 0.0

    print(f"\n")
    if label:
        print(f"  {label}")
    print(f"\n{'№':>3}  {'Ист':^5}  {'Лучш':^5}  {'Топ-5 гипотез'}")
    print("-" * 64)
    for i, hyps in enumerate(results):
        true_ch = true_text[i] if i < len(true_text) else "?"
        top5 = ", ".join(f"({c},{s:.3f})" for c, s in hyps[:5])
        best = hyps[0][0]
        mark = "OK" if best == true_ch else "!!"
        print(f"{i+1:>3}  {true_ch:^5}  [{mark}] {top5}")

    print()
    print(f"  Лучшие гипотезы : «{recognized}»")
    print(f"  Эталонный текст : «{true_text}»")
    print(f"  Ошибок          : {errors}")
    print(f"  Верно           : {correct}/{len(true_text)} = {accuracy:.1f}%")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Эталонный текст: «{true_text}»\n")
        f.write("Гипотезы:\n\n")
        for i, hyps in enumerate(results):
            hyp_str = ", ".join(f'("{c}", {s:.4f})' for c, s in hyps)
            f.write(f"{i+1}: [{hyp_str}]\n")
        f.write(f"Лучшие гипотезы: «{recognized}»\n")
        f.write(f"Эталонный текст: «{true_text}»\n")
        f.write(f"Ошибок: {errors}\n")
        f.write(f"Верно распознано: {correct}/{len(true_text)} = {accuracy:.1f}%\n")

    return recognized, errors, accuracy


def plot_heatmap(results, true_text, out_path, title=""):
    n = min(len(results), len(true_text))
    alpha_idx = {c: i for i, c in enumerate(ALPHABET)}
    mat = np.zeros((n, len(ALPHABET)))
    for i, hyps in enumerate(results[:n]):
        for (c, s) in hyps:
            if c in alpha_idx:
                mat[i, alpha_idx[c]] = s

    fig, ax = plt.subplots(figsize=(max(16, len(ALPHABET)*0.45), max(5, n*0.5)))
    im = ax.imshow(mat, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(len(ALPHABET)))
    ax.set_xticklabels(ALPHABET, fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{i+1}: {true_text[i]}" for i in range(n)], fontsize=9)
    ax.set_title(title or "Тепловая карта мер близости")
    ax.set_xlabel("Алфавит")
    ax.set_ylabel("Символы распознаваемого текста")
    plt.colorbar(im, ax=ax, label="Мера близости")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_comparison(res_main, res_exp, true_text, out_path):
    n = len(true_text)
    idx = range(n)

    def get_data(results):
        sims = [results[i][0][1] if i < len(results) else 0 for i in idx]
        cols = ['#2ecc71' if (i < len(results) and results[i][0][0] == true_text[i])
                else '#e74c3c' for i in idx]
        return sims, cols

    sm, cm = get_data(res_main)
    se, ce = get_data(res_exp)

    fig, axes = plt.subplots(2, 1, figsize=(max(12, n*0.6), 9))
    for ax, sims, cols, fs, lbl in [
        (axes[0], sm, cm, FONT_SIZE,     f"Основной шрифт ({FONT_SIZE}pt)"),
        (axes[1], se, ce, FONT_SIZE_EXP, f"Эксперимент ({FONT_SIZE_EXP}pt, база {FONT_SIZE}pt)"),
    ]:
        bars = ax.bar(idx, sims, color=cols, alpha=0.85, edgecolor='white', linewidth=0.5)
        ax.set_xticks(list(idx))
        ax.set_xticklabels(list(true_text), fontsize=11)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Мера близости")
        ax.set_title(f"{lbl}  (зелёный - верно, красный - ошибка)")
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
        for bar, s in zip(bars, sims):
            ax.text(bar.get_x() + bar.get_width()/2, s + 0.01,
                    f"{s:.2f}", ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def plot_feature_space(db, symbols, true_text, out_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    for char, fv in db.items():
        ax.scatter(fv[1], fv[2], c='lightblue', s=60, zorder=2)
        ax.annotate(char, (fv[1], fv[2]), fontsize=8, ha='center', va='bottom', color='steelblue')
    cmap = plt.cm.tab10(np.linspace(0, 1, len(symbols)))
    for i, sym in enumerate(symbols):
        fv = extract_features(sym)
        true_ch = true_text[i] if i < len(true_text) else "?"
        ax.scatter(fv[1], fv[2], c=[cmap[i]], s=130, marker='*',
                   zorder=3, edgecolors='black', linewidths=0.5)
        ax.annotate(f"↑{true_ch}", (fv[1], fv[2]+0.005), fontsize=9, color='darkred')
    ax.set_xlabel("cx — нормир. координата центра тяжести по X")
    ax.set_ylabel("cy — нормир. координата центра тяжести по Y")
    ax.set_title("Признаковое пространство cx–cy\n"
                 "(голубые кружки - алфавит, звёзды - распознаваемые символы)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    db = build_alphabet_db(FONT_SIZE)

    gray_main, path_main = generate_text_image(RECOGNIZED_TEXT, FONT_SIZE)
    print(f"    Текст: «{RECOGNIZED_TEXT}»")
    print(f"    Изображение: {path_main}")
    symbols_main = extract_symbols_from_image(gray_main)
    print(f"    Обнаружено символов: {len(symbols_main)}")

    results_main = classify(symbols_main, db)

    rec_main, err_main, acc_main = print_and_save_results(
        results_main, RECOGNIZED_TEXT,
        os.path.join(OUTPUT_DIR, "hypotheses_main.txt"),
        label=f"Основной шрифт {FONT_SIZE}pt"
    )

    plot_heatmap(results_main, RECOGNIZED_TEXT,
                 os.path.join(OUTPUT_DIR, "heatmap_main.png"),
                 f"Тепловая карта мер близости (шрифт {FONT_SIZE}pt)")

    plot_feature_space(db, symbols_main, RECOGNIZED_TEXT,
                       os.path.join(OUTPUT_DIR, "feature_space.png"))

    print(f"\nЭксперимент: генерация изображения шрифтом {FONT_SIZE_EXP}pt")
    gray_exp, path_exp = generate_text_image(RECOGNIZED_TEXT, FONT_SIZE_EXP)
    print(f"    Изображение: {path_exp}")
    symbols_exp = extract_symbols_from_image(gray_exp)
    print(f"    Обнаружено символов: {len(symbols_exp)}")

    results_exp = classify(symbols_exp, db)

    rec_exp, err_exp, acc_exp = print_and_save_results(
        results_exp, RECOGNIZED_TEXT,
        os.path.join(OUTPUT_DIR, "hypotheses_experiment.txt"),
        label=f"Эксперимент: {FONT_SIZE_EXP}pt, база {FONT_SIZE}pt"
    )

    plot_heatmap(results_exp, RECOGNIZED_TEXT,
                 os.path.join(OUTPUT_DIR, "heatmap_experiment.png"),
                 f"Тепловая карта мер близости ({FONT_SIZE_EXP}pt → база {FONT_SIZE}pt)")

    plot_comparison(results_main, results_exp, RECOGNIZED_TEXT,
                    os.path.join(OUTPUT_DIR, "comparison.png"))

    print("\n  Итоги:")
    print(f"  Основной шрифт {FONT_SIZE}pt:")
    print(f"    Распознано: «{rec_main}»")
    print(f"    Ошибок: {err_main}   Точность: {acc_main:.1f}%")
    print(f"  Эксперимент ({FONT_SIZE_EXP}pt, база {FONT_SIZE}pt):")
    print(f"    Распознано: «{rec_exp}»")
    print(f"    Ошибок: {err_exp}   Точность: {acc_exp:.1f}%")
    delta = acc_main - acc_exp
    print(f"  Изменение точности: {delta:-.1f}%")


if __name__ == "__main__":
    main()