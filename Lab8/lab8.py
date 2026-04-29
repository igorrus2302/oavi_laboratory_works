import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


GAMMA = 0.5
C_POWER = 1.0
F0 = 0.0

IMAGE_DIR = "images"
IMAGE_FILES = {
    "regular": os.path.join(IMAGE_DIR, "regular.png"),
    "low":     os.path.join(IMAGE_DIR, "low.png"),
    "high":    os.path.join(IMAGE_DIR, "high.png"),
}


def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def rgb_to_gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float64)


def rgb_to_hsl(rgb):
    r = rgb[:, :, 0] / 255.0
    g = rgb[:, :, 1] / 255.0
    b = rgb[:, :, 2] / 255.0

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    L = (cmax + cmin) / 2.0

    S = np.where(delta == 0, 0.0, delta / (1.0 - np.abs(2 * L - 1)))
    S = np.clip(S, 0, 1)

    H = np.zeros_like(L)
    mask_r = (cmax == r) & (delta != 0)
    mask_g = (cmax == g) & (delta != 0)
    mask_b = (cmax == b) & (delta != 0)
    H[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    H[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    H[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

    return H, S, L


def hsl_to_rgb(H, S, L):
    C = (1.0 - np.abs(2 * L - 1)) * S
    X = C * (1.0 - np.abs((H / 60.0) % 2 - 1))
    m = L - C / 2.0

    R1 = np.zeros_like(L)
    G1 = np.zeros_like(L)
    B1 = np.zeros_like(L)

    masks = [
        (H < 60),
        (H >= 60) & (H < 120),
        (H >= 120) & (H < 180),
        (H >= 180) & (H < 240),
        (H >= 240) & (H < 300),
        (H >= 300),
    ]
    vals = [
        (C, X, np.zeros_like(L)),
        (X, C, np.zeros_like(L)),
        (np.zeros_like(L), C, X),
        (np.zeros_like(L), X, C),
        (X, np.zeros_like(L), C),
        (C, np.zeros_like(L), X),
    ]
    for mask, (r, g, b) in zip(masks, vals):
        R1 = np.where(mask, r, R1)
        G1 = np.where(mask, g, G1)
        B1 = np.where(mask, b, B1)

    R = np.clip((R1 + m) * 255, 0, 255).astype(np.uint8)
    G = np.clip((G1 + m) * 255, 0, 255).astype(np.uint8)
    B = np.clip((B1 + m) * 255, 0, 255).astype(np.uint8)

    return np.stack([R, G, B], axis=-1)


def power_transform(L_channel, gamma=GAMMA, c=C_POWER, f0=F0):
    f = np.clip(L_channel + f0, 0, None)
    g = c * np.power(f, gamma)
    g_min, g_max = g.min(), g.max()
    if g_max - g_min > 1e-8:
        g = (g - g_min) / (g_max - g_min)
    return np.clip(g, 0, 1)


def compute_lbp(gray_img, radius=1):
    img = gray_img.astype(np.float64)
    rows, cols = img.shape
    lbp = np.zeros((rows, cols), dtype=np.uint8)

    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    neighbors = []
    for angle in angles:
        rad = np.deg2rad(angle)
        dr = -radius * np.sin(rad)
        dc =  radius * np.cos(rad)
        neighbors.append((dr, dc))

    for i in range(radius, rows - radius):
        for j in range(radius, cols - radius):
            center = img[i, j]
            code = 0
            for bit, (dr, dc) in enumerate(neighbors):
                ni = int(round(i + dr))
                nj = int(round(j + dc))
                ni = np.clip(ni, 0, rows - 1)
                nj = np.clip(nj, 0, cols - 1)
                if img[ni, nj] >= center:
                    code |= (1 << bit)
            lbp[i, j] = code

    return lbp


def compute_lbp_fast(gray_img):
    img = gray_img.astype(np.float64)
    rows, cols = img.shape

    p = np.pad(img, 1, mode='edge')
    center = p[1:rows+1, 1:cols+1]

    offsets = [
        (0,  2),
        (0,  2),
    ]
    neighbor_positions = [
        (1, 2),
        (0, 2),
        (0, 1),
        (0, 0),
        (1, 0),
        (2, 0),
        (2, 1),
        (2, 2),
    ]

    lbp = np.zeros((rows, cols), dtype=np.uint8)
    for bit, (nr, nc) in enumerate(neighbor_positions):
        neighbor = p[nr:nr+rows, nc:nc+cols]
        lbp += ((neighbor >= center).astype(np.uint8)) << bit

    return lbp


def lbp_histogram(lbp_map, n_bins=256, normalize=True):
    hist, _ = np.histogram(lbp_map.ravel(), bins=n_bins, range=(0, 256))
    if normalize:
        hist = hist / hist.sum()
    return hist


def brightness_histogram(channel_0_255, n_bins=256):
    hist, edges = np.histogram(channel_0_255.ravel(), bins=n_bins, range=(0, 256))
    hist = hist / hist.sum()
    return hist, edges[:-1]


def lbp_features_summary(hist):
    h = hist / (hist.sum() + 1e-12)
    mean_val   = np.sum(np.arange(len(h)) * h)
    std_val    = np.sqrt(np.sum((np.arange(len(h)) - mean_val)**2 * h))
    energy     = np.sum(h**2)
    entropy    = -np.sum(h * np.log2(h + 1e-12))
    uniformity = np.max(h)
    return {
        "mean":      mean_val,
        "std":       std_val,
        "energy":    energy,
        "entropy":   entropy,
        "uniformity": uniformity,
    }


def process_image(name, path):
    print("\n")
    print(f"  Изображение: {name} ")

    rgb = load_image(path)
    H_ch, S_ch, L_ch = rgb_to_hsl(rgb)
    gray = (L_ch * 255).astype(np.uint8)

    lbp_orig = compute_lbp_fast(gray.astype(np.float64))
    hist_lbp_orig = lbp_histogram(lbp_orig)
    feat_orig = lbp_features_summary(hist_lbp_orig)

    print(f"\n Исходное ")
    for k, v in feat_orig.items():
        print(f"  {k:12s}: {v:.4f}")

    L_contrast = power_transform(L_ch, gamma=GAMMA, c=C_POWER, f0=F0)

    rgb_contrast = hsl_to_rgb(H_ch, S_ch, L_contrast)
    gray_contrast = (L_contrast * 255).astype(np.uint8)

    lbp_contr = compute_lbp_fast(gray_contrast.astype(np.float64))
    hist_lbp_contr = lbp_histogram(lbp_contr)
    feat_contr = lbp_features_summary(hist_lbp_contr)

    print(f"\n После контрастирования (γ={GAMMA})")
    for k, v in feat_contr.items():
        print(f"  {k:12s}: {v:.4f}")

    hist_br_orig,  edges_orig  = brightness_histogram(gray)
    hist_br_contr, edges_contr = brightness_histogram(gray_contrast)

    return {
        "name":           name,
        "rgb_orig":       rgb,
        "rgb_contrast":   rgb_contrast,
        "gray_orig":      gray,
        "gray_contrast":  gray_contrast,
        "lbp_orig":       lbp_orig,
        "lbp_contr":      lbp_contr,
        "hist_lbp_orig":  hist_lbp_orig,
        "hist_lbp_contr": hist_lbp_contr,
        "feat_orig":      feat_orig,
        "feat_contr":     feat_contr,
        "hist_br_orig":   hist_br_orig,
        "hist_br_contr":  hist_br_contr,
        "edges":          edges_orig,
    }


def plot_results(results):
    name = results["name"]

    fig = plt.figure(figsize=(20, 18))
    fig.suptitle(f"LBP + Степенное преобразование\nИзображение: «{name}»",
                 fontsize=14, fontweight='bold')

    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.45, wspace=0.35)

    ax_rgb_o   = fig.add_subplot(gs[0, 0])
    ax_gray_o  = fig.add_subplot(gs[0, 1])
    ax_rgb_c   = fig.add_subplot(gs[0, 2])
    ax_gray_c  = fig.add_subplot(gs[0, 3])

    ax_rgb_o.imshow(results["rgb_orig"])
    ax_rgb_o.set_title("Исходное (RGB)", fontsize=9)
    ax_rgb_o.axis("off")

    ax_gray_o.imshow(results["gray_orig"], cmap="gray", vmin=0, vmax=255)
    ax_gray_o.set_title("Полутоновое (L)", fontsize=9)
    ax_gray_o.axis("off")

    ax_rgb_c.imshow(results["rgb_contrast"])
    ax_rgb_c.set_title(f"Контрастированное (RGB, γ={GAMMA})", fontsize=9)
    ax_rgb_c.axis("off")

    ax_gray_c.imshow(results["gray_contrast"], cmap="gray", vmin=0, vmax=255)
    ax_gray_c.set_title(f"Контрастир. полутоновое", fontsize=9)
    ax_gray_c.axis("off")

    ax_lbp_o  = fig.add_subplot(gs[1, 0:2])
    ax_lbp_c  = fig.add_subplot(gs[1, 2:4])

    im1 = ax_lbp_o.imshow(results["lbp_orig"], cmap="jet", vmin=0, vmax=255)
    ax_lbp_o.set_title("LBP-карта (исходное)", fontsize=9)
    ax_lbp_o.axis("off")
    plt.colorbar(im1, ax=ax_lbp_o, fraction=0.046, pad=0.04)

    im2 = ax_lbp_c.imshow(results["lbp_contr"], cmap="jet", vmin=0, vmax=255)
    ax_lbp_c.set_title("LBP-карта (контрастированное)", fontsize=9)
    ax_lbp_c.axis("off")
    plt.colorbar(im2, ax=ax_lbp_c, fraction=0.046, pad=0.04)

    ax_hlbp_o = fig.add_subplot(gs[2, 0:2])
    ax_hlbp_c = fig.add_subplot(gs[2, 2:4])

    bins = np.arange(256)
    ax_hlbp_o.bar(bins, results["hist_lbp_orig"], width=1.0, color="steelblue", alpha=0.8)
    ax_hlbp_o.set_title("H(LBP) — исходное", fontsize=9)
    ax_hlbp_o.set_xlabel("LBP-код", fontsize=8)
    ax_hlbp_o.set_ylabel("Нормир. частота", fontsize=8)
    ax_hlbp_o.tick_params(labelsize=7)

    ax_hlbp_c.bar(bins, results["hist_lbp_contr"], width=1.0, color="darkorange", alpha=0.8)
    ax_hlbp_c.set_title("H(LBP) — контрастированное", fontsize=9)
    ax_hlbp_c.set_xlabel("LBP-код", fontsize=8)
    ax_hlbp_c.set_ylabel("Нормир. частота", fontsize=8)
    ax_hlbp_c.tick_params(labelsize=7)

    ax_br_o  = fig.add_subplot(gs[3, 0:2])
    ax_feat  = fig.add_subplot(gs[3, 2:4])

    edges = results["edges"]
    ax_br_o.bar(edges, results["hist_br_orig"],  width=1.0,
                color="steelblue", alpha=0.6, label="До")
    ax_br_o.bar(edges, results["hist_br_contr"], width=1.0,
                color="darkorange", alpha=0.6, label="После")
    ax_br_o.set_title("Гистограмма яркости (до / после)", fontsize=9)
    ax_br_o.set_xlabel("Яркость", fontsize=8)
    ax_br_o.set_ylabel("Нормир. частота", fontsize=8)
    ax_br_o.legend(fontsize=8)
    ax_br_o.tick_params(labelsize=7)

    feat_names = list(results["feat_orig"].keys())
    col_labels = ["Признак", "До", "После", "Δ"]
    table_data = []
    for k in feat_names:
        v_o = results["feat_orig"][k]
        v_c = results["feat_contr"][k]
        table_data.append([k, f"{v_o:.4f}", f"{v_c:.4f}", f"{v_c - v_o:+.4f}"])

    ax_feat.axis("off")
    tbl = ax_feat.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.6)
    ax_feat.set_title("Сравнение LBP-признаков", fontsize=9, pad=12)

    out_path = f"lbp_{name}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_combined_lbp(all_results):
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle("Сравнение H(LBP) по всем изображениям\n"
                 "",
                 fontsize=13, fontweight="bold")

    colors = {"regular": "steelblue", "low": "seagreen", "high": "crimson"}
    bins = np.arange(256)

    for row, res in enumerate(all_results):
        name = res["name"]
        c = colors.get(name, "gray")

        ax_o = axes[row, 0]
        ax_c = axes[row, 1]

        ax_o.bar(bins, res["hist_lbp_orig"],  width=1, color=c, alpha=0.8)
        ax_o.set_title(f"«{name}» — H(LBP) исходное", fontsize=9)
        ax_o.set_xlabel("LBP-код"); ax_o.set_ylabel("Нормир. частота")

        ax_c.bar(bins, res["hist_lbp_contr"], width=1, color=c, alpha=0.5)
        ax_c.bar(bins, res["hist_lbp_orig"],  width=1, color=c, alpha=0.3,
                 label="исходное")
        ax_c.set_title(f"«{name}» — H(LBP) после γ-коррекции (γ={GAMMA})", fontsize=9)
        ax_c.set_xlabel("LBP-код"); ax_c.set_ylabel("Нормир. частота")

    plt.tight_layout()
    out_path = "combined.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":

    found_images = {}
    for key, path in IMAGE_FILES.items():
        if os.path.exists(path):
            found_images[key] = path


    all_results = []
    output_files = []

    for name, path in found_images.items():
        try:
            res = process_image(name, path)
            all_results.append(res)
            out = plot_results(res)
            output_files.append(out)
        except Exception as e:
            print(f" Ошибка при обработке {name}: {e}")
            import traceback; traceback.print_exc()

    if len(all_results) > 1:
        out = plot_combined_lbp(all_results)
        output_files.append(out)

    print("\n Готово")