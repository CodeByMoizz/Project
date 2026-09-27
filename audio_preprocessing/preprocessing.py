import os
import sys

import numpy as np
import librosa
import soundfile as sf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import (
    APPLY_NOISE_REDUCTION,
    SEGMENT_DURATION_SEC,
    SILENCE_TOP_DB,
    TARGET_SAMPLE_RATE,
)

PAD_TRUNCATE_DURATION_SEC = SEGMENT_DURATION_SEC
OUTPUT_FORMAT = "wav"

MAX_SEGMENTS = 300


def load_audio(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        samples, sr = librosa.load(file_path, sr=None, mono=False)
    except Exception as e:
        raise ValueError(f"Could not load audio file {file_path}: {e}")

    if samples is None or samples.size == 0:
        raise ValueError(f"Audio file has no samples: {file_path}")

    return samples, sr


def convert_to_mono(samples):
    samples = np.asarray(samples, dtype=np.float32)

    if samples.ndim == 1:
        return samples

    return np.mean(samples, axis=0).astype(np.float32)


def resample_audio(samples, original_sr, target_sr=TARGET_SAMPLE_RATE):
    if original_sr == target_sr:
        return samples, target_sr

    resampled = librosa.resample(samples, orig_sr=original_sr, target_sr=target_sr)
    return resampled, target_sr


def normalize_audio(samples):
    peak = np.max(np.abs(samples))

    if peak < 1e-6:
        return samples

    return (samples / peak).astype(np.float32)


def trim_silence(samples, top_db=SILENCE_TOP_DB):
    if len(samples) == 0:
        return samples

    trimmed, _ = librosa.effects.trim(samples, top_db=top_db)

    if len(trimmed) == 0:
        return samples

    return trimmed


def reduce_noise(samples, sr, noise_estimate_sec=0.5):
    # Simple spectral floor subtraction using the start of the clip as the
    # noise estimate. Skipped when the start is not actually quieter than the
    # rest, because subtracting then removes part of the event itself.
    noise_samples_count = int(noise_estimate_sec * sr)

    if noise_samples_count < 1 or noise_samples_count >= len(samples):
        return samples

    noise_clip = samples[:noise_samples_count]
    noise_level = float(np.mean(np.abs(noise_clip)))
    overall_level = float(np.mean(np.abs(samples)))

    if noise_level >= overall_level * 0.9:
        return samples

    reduced = np.sign(samples) * np.maximum(np.abs(samples) - noise_level, 0)
    return reduced.astype(np.float32)


def segment_audio(samples, sr, segment_duration=SEGMENT_DURATION_SEC):
    segment_len = int(segment_duration * sr)

    if segment_len <= 0:
        raise ValueError("segment_duration must be greater than 0")

    if len(samples) <= segment_len:
        return [samples], [(0.0, len(samples) / sr)]

    segments = []
    segment_info = []

    start = 0
    while start < len(samples) and len(segments) < MAX_SEGMENTS:
        end = start + segment_len
        segments.append(samples[start:end])
        segment_info.append((start / sr, min(end, len(samples)) / sr))
        start = end

    return segments, segment_info


def pad_or_truncate(samples, sr, target_duration=PAD_TRUNCATE_DURATION_SEC):
    target_len = int(target_duration * sr)

    if len(samples) == 0:
        return np.zeros(target_len, dtype=np.float32)

    if len(samples) < target_len:
        pad_amount = target_len - len(samples)
        return np.pad(samples, (0, pad_amount), mode="constant").astype(np.float32)

    return samples[:target_len].astype(np.float32)


def save_processed_audio(samples, sr, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    sf.write(output_path, samples, sr, format=OUTPUT_FORMAT.upper())
    return output_path


# The one preprocessing entry point. Everything that needs model-ready audio
# calls this, so uploads, live windows and dataset files are treated the same.
def prepare_segments(file_path, apply_noise_reduction=APPLY_NOISE_REDUCTION):
    samples, sr = load_audio(file_path)
    return prepare_segments_from_samples(samples, sr, apply_noise_reduction)


def prepare_segments_from_samples(samples, sr, apply_noise_reduction=APPLY_NOISE_REDUCTION):
    samples = convert_to_mono(samples)
    samples = np.nan_to_num(samples)
    samples, sr = resample_audio(samples, sr)
    samples = normalize_audio(samples)
    samples = trim_silence(samples)

    if apply_noise_reduction:
        samples = reduce_noise(samples, sr)

    segments, segment_info = segment_audio(samples, sr)

    fixed_segments = [pad_or_truncate(segment, sr) for segment in segments]

    return fixed_segments, sr, segment_info


def preprocess_pipeline(file_path, output_dir="data/processed", apply_noise_reduction=APPLY_NOISE_REDUCTION):
    segments, sr, _ = prepare_segments(file_path, apply_noise_reduction)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    saved_paths = []

    for i, segment in enumerate(segments):
        out_name = f"{base_name}_seg{i}.{OUTPUT_FORMAT}"
        out_path = os.path.join(output_dir, out_name)
        saved_paths.append(save_processed_audio(segment, sr, out_path))

    return saved_paths


if __name__ == "__main__":
    sample_file = os.path.join("sample_audio", "test1.wav")

    if os.path.exists(sample_file):
        for p in preprocess_pipeline(sample_file):
            print("saved", p)
    else:
        print(f"Sample file not found: {sample_file}")
