import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft, get_window
import soundfile as sf
import pandas as pd
import os


INPUT_FILE = "sound.wav"
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_DURATION = 0.05
OVERLAP = 0.75
DT = 0.1
DF = 50

signal, sr = sf.read(INPUT_FILE)

if signal.ndim > 1:
    signal = signal.mean(axis=1)

N = len(signal)
duration = N / sr

nperseg = int(WINDOW_DURATION * sr)
noverlap = int(nperseg * OVERLAP)
window = get_window('hann', nperseg)

f, t, Zxx = stft(signal, fs=sr, window=window,
                 nperseg=nperseg, noverlap=noverlap)

magnitude = np.abs(Zxx)

plt.figure(figsize=(10, 5))
plt.pcolormesh(t, f, 20 * np.log10(magnitude + 1e-10), shading='gouraud')
plt.yscale('log')
plt.title("Spectrogram before")
plt.xlabel("Time, s")
plt.ylabel("Frequency, Hz")
plt.colorbar(label="dB")
plt.savefig(f"{OUTPUT_DIR}/spectrogram_before.png")
plt.close()

energy = np.sum(magnitude ** 2, axis=0)

threshold = np.percentile(energy, 20)
noise_frames = magnitude[:, energy < threshold]

noise_spectrum = np.mean(noise_frames, axis=1, keepdims=True)

k = 1.0

clean_magnitude = magnitude - k * noise_spectrum
clean_magnitude = np.maximum(clean_magnitude, 0)

Zxx_clean = clean_magnitude * np.exp(1j * np.angle(Zxx))

_, signal_clean = istft(Zxx_clean, fs=sr,
                        window=window,
                        nperseg=nperseg,
                        noverlap=noverlap)

signal_clean = signal_clean[:len(signal)]

sf.write(f"{OUTPUT_DIR}/denoised.wav", signal_clean, sr)

f2, t2, Zxx2 = stft(signal_clean, fs=sr,
                    window=window,
                    nperseg=nperseg,
                    noverlap=noverlap)

plt.figure(figsize=(10, 5))
plt.pcolormesh(t2, f2, 20 * np.log10(np.abs(Zxx2) + 1e-10), shading='gouraud')
plt.yscale('log')
plt.title("Spectrogram after")
plt.xlabel("Time, s")
plt.ylabel("Frequency, Hz")
plt.colorbar(label="dB")
plt.savefig(f"{OUTPUT_DIR}/spectrogram_after.png")
plt.close()

plt.figure()
plt.plot(signal)
plt.title("Waveform before")
plt.savefig(f"{OUTPUT_DIR}/waveform.png")
plt.close()

plt.figure()
plt.plot(signal_clean)
plt.title("Waveform after")
plt.savefig(f"{OUTPUT_DIR}/waveform_denoised.png")
plt.close()

noise = signal - signal_clean

def snr(signal, noise):
    return 10 * np.log10(np.sum(signal ** 2) / np.sum(noise ** 2))


snr_before = snr(signal, noise)
snr_after = snr(signal_clean, signal - signal_clean)

time_bins = np.arange(0, duration, DT)
freq_bins = np.arange(0, sr / 2, DF)

peaks = []

for i in range(len(time_bins) - 1):
    t_mask = (t >= time_bins[i]) & (t < time_bins[i + 1])

    for j in range(len(freq_bins) - 1):
        f_mask = (f >= freq_bins[j]) & (f < freq_bins[j + 1])

        if np.any(t_mask) and np.any(f_mask):
            block = magnitude[np.ix_(f_mask, t_mask)]
            energy_block = np.sum(block ** 2)

            peaks.append([
                time_bins[i],
                time_bins[i + 1],
                freq_bins[j],
                freq_bins[j + 1],
                energy_block
            ])

peaks = sorted(peaks, key=lambda x: x[4], reverse=True)

df = pd.DataFrame(peaks, columns=["t1", "t2", "f1", "f2", "E"])
df.to_csv(f"{OUTPUT_DIR}/energy_peaks.csv", index=False)

print("SNR before:", snr_before)
print("SNR after:", snr_after)
print("Top-5 peaks:")
for p in peaks[:5]:
    print(p)