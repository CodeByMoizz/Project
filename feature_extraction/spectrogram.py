import os
import sys

import librosa
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audio_preprocessing.preprocessing import prepare_segments
from config.config import SEGMENT_DURATION_SEC, TARGET_SAMPLE_RATE

# The CNN reads a log-mel spectrogram instead of the 215 summary values the other
# models use. Everything that needs a spectrogram calls this module, so training
# and inference cannot drift apart, the same way features.py works for the
# summary features.

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
FMIN = 20
FMAX = TARGET_SAMPLE_RATE // 2

# Decibels are clipped to this range and then scaled to roughly 0..1. A fixed
# range is used on purpose: it means the CNN needs no scaler file, so a
# spectrogram built today matches one built during training.
DB_FLOOR = -80.0

FRAME_COUNT = 1 + int(SEGMENT_DURATION_SEC * TARGET_SAMPLE_RATE) // HOP_LENGTH
INPUT_SHAPE = (N_MELS, FRAME_COUNT, 1)


def build_spectrogram(samples, sr=TARGET_SAMPLE_RATE):
    samples = np.asarray(samples, dtype=np.float32)

    if samples.ndim > 1:
        samples = np.mean(samples, axis=0)

    samples = np.nan_to_num(samples)

    mel = librosa.feature.melspectrogram(
        y=samples,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        fmax=FMAX,
    )

    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=abs(DB_FLOOR))

    scaled = (mel_db - DB_FLOOR) / abs(DB_FLOOR)
    scaled = np.clip(scaled, 0.0, 1.0)

    scaled = fit_frames(scaled)

    return scaled.astype(np.float32)[..., np.newaxis]


# A short segment can come back a frame or two shorter, and the CNN needs one
# fixed shape, so the frame count is padded or trimmed.
def fit_frames(spectrogram, frames=FRAME_COUNT):
    current = spectrogram.shape[1]

    if current == frames:
        return spectrogram

    if current < frames:
        return np.pad(spectrogram, ((0, 0), (0, frames - current)), mode="constant")

    return spectrogram[:, :frames]


# A file is preprocessed first, so a dataset clip and an uploaded clip travel
# the exact same path.
def spectrogram_from_file(file_path):
    segments, sr, _ = prepare_segments(file_path)

    if not segments:
        raise ValueError(f"No usable audio in {file_path}")

    return build_spectrogram(segments[0], sr)
