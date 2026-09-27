import os
import sys

import librosa
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audio_preprocessing.preprocessing import prepare_segments
from config.config import TARGET_SAMPLE_RATE

N_MFCC = 20
N_MELS = 40
N_CONTRAST_BANDS = 7


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


# How fast the MFCCs change over time. This is what separates a short burst like
# a gunshot from a steady sound like a siren, which look similar on averages
# alone.
def extract_mfcc_delta(y, sr, n_mfcc=N_MFCC):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    return get_mean_std(delta)


def extract_mel_spectrogram(y, sr, n_mels=N_MELS):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel)
    return get_mean_std(mel_db)


def extract_chroma(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return get_mean_std(chroma)


# The gap between peaks and valleys in each frequency band. Useful for telling
# tonal sounds such as a horn from broadband ones such as breaking glass.
def extract_spectral_contrast(y, sr):
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    return get_mean_std(contrast)


def extract_zcr(y):
    zcr = librosa.feature.zero_crossing_rate(y)
    return float(np.mean(zcr)), float(np.std(zcr))


def extract_rms(y):
    rms = librosa.feature.rms(y=y)[0]
    mean = float(np.mean(rms))
    peak = float(np.max(rms))

    # Crest factor: how far the loudest moment stands above the average one.
    crest = peak / (mean + 1e-9)

    return mean, float(np.std(rms)), peak, crest


def extract_spectral_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return float(np.mean(centroid)), float(np.std(centroid))


def extract_spectral_bandwidth(y, sr):
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    return float(np.mean(bandwidth)), float(np.std(bandwidth))


def extract_spectral_rolloff(y, sr):
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    return float(np.mean(rolloff)), float(np.std(rolloff))


def extract_spectral_flatness(y):
    flatness = librosa.feature.spectral_flatness(y=y)
    return float(np.mean(flatness)), float(np.std(flatness))


def extract_onset_strength(y, sr):
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    return float(np.mean(onset)), float(np.std(onset)), float(np.max(onset))


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

    delta_mean, delta_std = extract_mfcc_delta(y, sr)
    add_to_dict(features, "mfcc_delta", delta_mean, delta_std)

    mel_mean, mel_std = extract_mel_spectrogram(y, sr)
    add_to_dict(features, "mel", mel_mean, mel_std)

    chroma_mean, chroma_std = extract_chroma(y, sr)
    add_to_dict(features, "chroma", chroma_mean, chroma_std)

    contrast_mean, contrast_std = extract_spectral_contrast(y, sr)
    add_to_dict(features, "contrast", contrast_mean, contrast_std)

    zcr_mean, zcr_std = extract_zcr(y)
    features["zcr_mean"] = zcr_mean
    features["zcr_std"] = zcr_std

    rms_mean, rms_std, rms_peak, rms_crest = extract_rms(y)
    features["rms_mean"] = rms_mean
    features["rms_std"] = rms_std
    features["rms_peak"] = rms_peak
    features["rms_crest"] = rms_crest

    centroid_mean, centroid_std = extract_spectral_centroid(y, sr)
    features["spectral_centroid_mean"] = centroid_mean
    features["spectral_centroid_std"] = centroid_std

    bandwidth_mean, bandwidth_std = extract_spectral_bandwidth(y, sr)
    features["spectral_bandwidth_mean"] = bandwidth_mean
    features["spectral_bandwidth_std"] = bandwidth_std

    rolloff_mean, rolloff_std = extract_spectral_rolloff(y, sr)
    features["spectral_rolloff_mean"] = rolloff_mean
    features["spectral_rolloff_std"] = rolloff_std

    flatness_mean, flatness_std = extract_spectral_flatness(y)
    features["spectral_flatness_mean"] = flatness_mean
    features["spectral_flatness_std"] = flatness_std

    onset_mean, onset_std, onset_peak = extract_onset_strength(y, sr)
    features["onset_strength_mean"] = onset_mean
    features["onset_strength_std"] = onset_std
    features["onset_strength_peak"] = onset_peak

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
