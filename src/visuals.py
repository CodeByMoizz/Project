import os
import sys

import matplotlib

matplotlib.use("Agg")

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import VISUAL_DIR


def visual_dir():
    os.makedirs(VISUAL_DIR, exist_ok=True)
    return VISUAL_DIR


def save_waveform(samples, sr, name):
    samples = np.nan_to_num(np.asarray(samples, dtype=np.float32))

    out_path = visual_dir() / f"{name}_waveform.png"

    figure = plt.figure(figsize=(8, 2.2))
    try:
        librosa.display.waveshow(samples, sr=sr, color="#238b6c")
        plt.margins(x=0)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        figure.savefig(out_path, dpi=110)
    finally:
        plt.close(figure)

    return f"visuals/{out_path.name}"


def save_spectrogram(samples, sr, name):
    samples = np.nan_to_num(np.asarray(samples, dtype=np.float32))

    out_path = visual_dir() / f"{name}_spectrogram.png"

    figure = plt.figure(figsize=(8, 2.2))
    try:
        mel = librosa.feature.melspectrogram(y=samples, sr=sr, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.tight_layout()
        figure.savefig(out_path, dpi=110)
    finally:
        plt.close(figure)

    return f"visuals/{out_path.name}"


# Failing to draw a picture must never fail the analysis, so both are guarded.
def make_visuals(samples, sr, name):
    waveform_path = None
    spectrogram_path = None

    try:
        waveform_path = save_waveform(samples, sr, name)
    except Exception:
        waveform_path = None

    try:
        spectrogram_path = save_spectrogram(samples, sr, name)
    except Exception:
        spectrogram_path = None

    return waveform_path, spectrogram_path
