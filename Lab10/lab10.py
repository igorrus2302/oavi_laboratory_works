import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


TARGET_SR = 16000


def preprocess_audio(path):
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)

    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    y, _ = librosa.effects.trim(y, top_db=25)

    return y, TARGET_SR


def plot_spectrogram(y, sr, title, save_path):
    D = librosa.stft(y, window='hann')
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    plt.figure(figsize=(10, 5))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.T


def segment_audio(y, sr):
    intervals = librosa.effects.split(y, top_db=25)
    return intervals

def merge_segments(intervals, sr, min_duration=0.2, max_gap=0.1):
    merged = []

    current_start, current_end = intervals[0]

    for start, end in intervals[1:]:
        gap = (start - current_end) / sr
        duration = (current_end - current_start) / sr

        if gap < max_gap or duration < min_duration:
            current_end = end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    merged.append((current_start, current_end))

    return merged


def dtw_distance(a, b):
    dist = cdist(a, b, metric='euclidean')
    D = np.zeros_like(dist)

    D[0, 0] = dist[0, 0]

    for i in range(1, len(a)):
        D[i, 0] = dist[i, 0] + D[i-1, 0]

    for j in range(1, len(b)):
        D[0, j] = dist[0, j] + D[0, j-1]

    for i in range(1, len(a)):
        for j in range(1, len(b)):
            D[i, j] = dist[i, j] + min(
                D[i-1, j],
                D[i, j-1],
                D[i-1, j-1]
            )

    return D[-1, -1]


def load_templates(folder):
    templates = {}

    for file in os.listdir(folder):
        if file.endswith(".wav"):
            label = file.replace(".wav", "")
            path = os.path.join(folder, file)

            y, sr = preprocess_audio(path)
            templates[label] = extract_features(y, sr)

    return templates


def recognize_segment(segment, templates, sr):
    features = extract_features(segment, sr)

    best_label = None
    best_score = float('inf')

    for label, template in templates.items():
        score = dtw_distance(features, template)

        if score < best_score:
            best_score = score
            best_label = label

    return best_label, best_score


def recognize_phone(audio_path, templates):
    y, sr = preprocess_audio(audio_path)

    segments = segment_audio(y, sr)
    segments = merge_segments(segments, sr)

    print("Segments after merge:", len(segments))

    print("Segments found:", len(segments))

    result = []
    scores = []

    for start, end in segments:
        seg = y[start:end]

        label, score = recognize_segment(seg, templates, sr)

        result.append(label)
        scores.append(score)

    return result, scores


def evaluate(predicted, true):
    errors = sum(p != t for p, t in zip(predicted, true))
    accuracy = 1 - errors / len(true)
    return errors, accuracy


if __name__ == "__main__":

    templates = load_templates("audio/digits")

    y, sr = preprocess_audio("audio/phone.wav")

    print("Sample rate:", sr)

    plot_spectrogram(y, sr, "Телефон", "spectrogram.png")

    predicted, scores = recognize_phone("audio/phone.wav", templates)

    print("Распознано:", predicted)

    true = ["+", "7", "9", "8", "5", "2", "0", "4", "7", "6", "0", "4"]

    errors, acc = evaluate(predicted, true)

    print("Ошибки:", errors)
    print("Достоверность:", acc)