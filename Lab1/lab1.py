import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)

def save_image(arr, path):
    Image.fromarray(arr.astype(np.uint8)).save(path)

def show_images(images, titles, cols=3):
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(15, 5 * rows))

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)

        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        else:
            plt.imshow(img.astype(np.uint8))

        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def split_rgb(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    R_img = np.stack([R, np.zeros_like(R), np.zeros_like(R)], axis=2)
    G_img = np.stack([np.zeros_like(G), G, np.zeros_like(G)], axis=2)
    B_img = np.stack([np.zeros_like(B), np.zeros_like(B), B], axis=2)

    save_image(R_img, "R.png")
    save_image(G_img, "G.png")
    save_image(B_img, "B.png")

    return R, G, B, R_img, G_img, B_img

def rgb_to_hsi(img):
    img = img / 255.0
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    I = (R + G + B) / 3

    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-6

    theta = np.arccos(np.clip(num / den, -1, 1))
    H = np.where(B <= G, theta, 2 * np.pi - theta)
    H = H / (2 * np.pi)

    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 * min_rgb / (R + G + B + 1e-6))

    return H, S, I

def hsi_to_rgb(H, S, I):
    H = H * 2 * np.pi

    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)

    idx = (H >= 0) & (H < 2*np.pi/3)
    B[idx] = I[idx] * (1 - S[idx])
    R[idx] = I[idx] * (1 + (S[idx] * np.cos(H[idx])) /
                       (np.cos(np.pi/3 - H[idx]) + 1e-6))
    G[idx] = 3 * I[idx] - (R[idx] + B[idx])

    idx = (H >= 2*np.pi/3) & (H < 4*np.pi/3)
    H2 = H[idx] - 2*np.pi/3
    R[idx] = I[idx] * (1 - S[idx])
    G[idx] = I[idx] * (1 + (S[idx] * np.cos(H2)) /
                       (np.cos(np.pi/3 - H2) + 1e-6))
    B[idx] = 3 * I[idx] - (R[idx] + G[idx])

    idx = (H >= 4*np.pi/3) & (H < 2*np.pi)
    H3 = H[idx] - 4*np.pi/3
    G[idx] = I[idx] * (1 - S[idx])
    B[idx] = I[idx] * (1 + (S[idx] * np.cos(H3)) /
                       (np.cos(np.pi/3 - H3) + 1e-6))
    R[idx] = 3 * I[idx] - (G[idx] + B[idx])

    rgb = np.stack([R, G, B], axis=2)
    rgb = np.clip(rgb * 255, 0, 255)

    return rgb

def invert_intensity(img):
    H, S, I = rgb_to_hsi(img)

    I_inv = 1 - I

    result = hsi_to_rgb(H, S, I_inv)

    return result

def upscale(img, M):
    h, w, c = img.shape
    result = np.zeros((h * M, w * M, c))

    for i in range(h * M):
        for j in range(w * M):
            result[i, j] = img[i // M, j // M]

    return result

def downscale(img, N):
    h, w, c = img.shape
    result = np.zeros((h // N, w // N, c))

    for i in range(h // N):
        for j in range(w // N):
            result[i, j] = img[i * N, j * N]

    return result

def resample_two_pass(img, M, N):
    return downscale(upscale(img, M), N)

def resample_one_pass(img, M, N):
    h, w, c = img.shape

    new_h = int(h * M / N)
    new_w = int(w * M / N)

    result = np.zeros((new_h, new_w, c))

    for i in range(new_h):
        for j in range(new_w):
            src_x = int(i * N / M)
            src_y = int(j * N / M)

            src_x = min(src_x, h - 1)
            src_y = min(src_y, w - 1)

            result[i, j] = img[src_x, src_y]

    return result


if __name__ == "__main__":
    img = load_image("picture.png")

    R, G, B, R_img, G_img, B_img = split_rgb(img)

    H, S, I = rgb_to_hsi(img)
    save_image(I * 255, "intensity.png")

    inv = invert_intensity(img)
    save_image(inv, "inverted.png")

    up = upscale(img, 2)
    save_image(up, "upscale.png")

    down = downscale(img, 2)
    save_image(down, "downscale.png")

    two_pass = resample_two_pass(img, 3, 2)
    save_image(two_pass, "two_pass.png")

    one_pass = resample_one_pass(img, 3, 2)
    save_image(one_pass, "one_pass.png")

    show_images(
        [img, R_img, G_img, B_img],
        ["Original", "Red", "Green", "Blue"]
    )

    show_images(
        [I, inv],
        ["Intensity (HSI)", "Inverted Intensity"]
    )

    show_images(
        [img, up, down, two_pass, one_pass],
        ["Original", "Upscale x2", "Downscale /2", "Two-pass (3/2)", "One-pass (3/2)"],
        cols=2
    )