import os
import sys

import librosa
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audio_preprocessing.preprocessing import prepare_segments
from config.config import TARGET_SAMPLE_RATE

N_MFCC = 13
N_MELS = 40
N_CHROMA = 12


def get_mean_std(feature_array):
    mean = np.mean(feature_array, axis=1)
    std = np.std(feature_array, axis=1)
    return mean, std


def add_to_dict(features_dict, name, mean, std):
    for i in range(len(mean)):
        features_dict[f"{name}_{i+1}_mean"] = float(mean[i])
        features_dict[f"{name}_{i+1}_std"] = float(std[i])


def extract_mfcc(y, sr, n_mfcc=N_MFCC):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return get_mean_std(mfcc)


def extract_mel_spectrogram(y, sr, n_mels=N_MELS):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel)
    return get_mean_std(mel_db)


def extract_chroma(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return get_mean_std(chroma)


def extract_zcr(y):
    zcr = librosa.feature.zero_crossing_rate(y)
    return float(np.mean(zcr)), float(np.std(zcr))


def extract_rms(y):
    rms = librosa.feature.rms(y=y)
    return float(np.mean(rms)), float(np.std(rms))


def extract_spectral_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return float(np.mean(centroid)), float(np.std(centroid))


def extract_spectral_bandwidth(y, sr):
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    return float(np.mean(bandwidth)), float(np.std(bandwidth))


def extract_spectral_rolloff(y, sr):
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    return float(np.mean(rolloff)), float(np.std(rolloff))


def extract_onset_strength(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    return float(np.mean(onset_env)), float(np.std(onset_env))


def extract_tempo(y, sr):
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # librosa returns a scalar on some versions and a 1-element array on others
    tempo = np.atleast_1d(tempo)
    if len(tempo) == 0:
        return 0.0
    return float(tempo[0])


# Every feature vector in the project is built here, so training and inference
# cannot drift apart. The samples must already be mono at TARGET_SAMPLE_RATE.
def extract_features_from_samples(y, sr=TARGET_SAMPLE_RATE):
    y = np.asarray(y, dtype=np.float32)

    if y.ndim > 1:
        y = np.mean(y, axis=0)

    y = np.nan_to_num(y)

    features = {}

    mfcc_mean, mfcc_std = extract_mfcc(y, sr)
    add_to_dict(features, "mfcc", mfcc_mean, mfcc_std)

    mel_mean, mel_std = extract_mel_spectrogram(y, sr)
    add_to_dict(features, "mel", mel_mean, mel_std)

    chroma_mean, chroma_std = extract_chroma(y, sr)
    add_to_dict(features, "chroma", chroma_mean, chroma_std)

    zcr_mean, zcr_std = extract_zcr(y)
    features["zcr_mean"] = zcr_mean
    features["zcr_std"] = zcr_std

    rms_mean, rms_std = extract_rms(y)
    features["rms_mean"] = rms_mean
    features["rms_std"] = rms_std

    centroid_mean, centroid_std = extract_spectral_centroid(y, sr)
    features["spectral_centroid_mean"] = centroid_mean
    features["spectral_centroid_std"] = centroid_std

    bandwidth_mean, bandwidth_std = extract_spectral_bandwidth(y, sr)
    features["spectral_bandwidth_mean"] = bandwidth_mean
    features["spectral_bandwidth_std"] = bandwidth_std

    rolloff_mean, rolloff_std = extract_spectral_rolloff(y, sr)
    features["spectral_rolloff_mean"] = rolloff_mean
    features["spectral_rolloff_std"] = rolloff_std

    onset_mean, onset_std = extract_onset_strength(y, sr)
    features["onset_strength_mean"] = onset_mean
    features["onset_strength_std"] = onset_std

    features["tempo"] = extract_tempo(y, sr)

    return features


def feature_names():
    silent_segment = np.zeros(int(TARGET_SAMPLE_RATE * 1.0), dtype=np.float32)
    return list(extract_features_from_samples(silent_segment).keys())


# The column order of a feature vector must never depend on dict ordering,
# otherwise the model receives its inputs shuffled.
def features_to_vector(features):
    return [float(features[name]) for name in FEATURE_NAMES]


# A file is preprocessed first, so a segment file from the dataset and an
# uploaded clip travel the exact same path.
def extract_all_features(file_path):
    segments, sr, _ = prepare_segments(file_path)

    if not segments:
        raise ValueError(f"No usable audio in {file_path}")

    return extract_features_from_samples(segments[0], sr)


FEATURE_NAMES = feature_names()
FEATURE_VECTOR_LENGTH = len(FEATURE_NAMES)
