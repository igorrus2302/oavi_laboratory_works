import librosa

y, sr = librosa.load("plus.wav", sr=None, mono=False)
print(y.shape)
print(sr)