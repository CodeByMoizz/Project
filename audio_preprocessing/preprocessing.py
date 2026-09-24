import os
import numpy as np
import librosa
import soundfile as sf

# project settings, can be changed later after dataset analysis
TARGET_SAMPLE_RATE = 16000
SILENCE_TOP_DB = 30
SEGMENT_DURATION_SEC = 2.0
PAD_TRUNCATE_DURATION_SEC = 2.0
OUTPUT_FORMAT = "wav"


def load_audio(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        samples, sr = librosa.load(file_path, sr=None, mono=False)
    except Exception as e:
        raise ValueError(f"Could not load audio file {file_path}: {e}")

    return samples, sr


def convert_to_mono(samples):
    if samples.ndim == 1:
        return samples

    mono_samples = np.mean(samples, axis=0)
    return mono_samples


def resample_audio(samples, original_sr, target_sr=TARGET_SAMPLE_RATE):
    if original_sr == target_sr:
        return samples, target_sr

    resampled = librosa.resample(samples, orig_sr=original_sr, target_sr=target_sr)
    return resampled, target_sr


def normalize_audio(samples):
    peak = np.max(np.abs(samples))

    if peak < 1e-6:
        return samples

    normalized = samples / peak
    return normalized


def trim_silence(samples, top_db=SILENCE_TOP_DB):
    trimmed, _ = librosa.effects.trim(samples, top_db=top_db)

    if len(trimmed) == 0:
        return samples

    return trimmed


def reduce_noise(samples, sr, noise_estimate_sec=0.5):
    # simple method, assumes start of clip is background noise
    noise_samples_count = int(noise_estimate_sec * sr)

    if noise_samples_count >= len(samples):
        return samples

    noise_clip = samples[:noise_samples_count]
    noise_level = np.mean(np.abs(noise_clip))

    reduced = np.sign(samples) * np.maximum(np.abs(samples) - noise_level, 0)
    return reduced


def segment_audio(samples, sr, segment_duration=SEGMENT_DURATION_SEC):
    segment_len = int(segment_duration * sr)

    if segment_len <= 0:
        raise ValueError("segment_duration must be greater than 0")

    if len(samples) <= segment_len:
        return [samples], [(0.0, len(samples) / sr)]

    segments = []
    segment_info = []

    start = 0
    while start < len(samples):
        end = start + segment_len
        segment = samples[start:end]
        segments.append(segment)
        segment_info.append((start / sr, min(end, len(samples)) / sr))
        start = end

    return segments, segment_info


def pad_or_truncate(samples, sr, target_duration=PAD_TRUNCATE_DURATION_SEC):
    target_len = int(target_duration * sr)

    if len(samples) == 0:
        return np.zeros(target_len)

    if len(samples) < target_len:
        pad_amount = target_len - len(samples)
        return np.pad(samples, (0, pad_amount), mode="constant")

    if len(samples) > target_len:
        return samples[:target_len]

    return samples


def save_processed_audio(samples, sr, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sf.write(output_path, samples, sr, format=OUTPUT_FORMAT.upper())
    return output_path


def preprocess_pipeline(file_path, output_dir="data/processed", apply_noise_reduction=True):
    samples, sr = load_audio(file_path)
    samples = convert_to_mono(samples)
    samples, sr = resample_audio(samples, sr)
    samples = normalize_audio(samples)
    samples = trim_silence(samples)

    if apply_noise_reduction:
        samples = reduce_noise(samples, sr)

    segments, segment_info = segment_audio(samples, sr)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    saved_paths = []

    for i, segment in enumerate(segments):
        fixed_segment = pad_or_truncate(segment, sr)
        out_name = f"{base_name}_seg{i}.{OUTPUT_FORMAT}"
        out_path = os.path.join(output_dir, out_name)
        saved_path = save_processed_audio(fixed_segment, sr, out_path)
        saved_paths.append(saved_path)

    return saved_paths


if __name__ == "__main__":
    sample_file = os.path.join("sample_audio", "test.wav")
    
    if os.path.exists(sample_file):
        result_paths = preprocess_pipeline(sample_file)
        print("Processed segments saved to:")
        for p in result_paths:
            print(" -", p)
    else:
        print(f"Sample file not found: {sample_file}")
        print("Put a test.wav file inside sample_audio/ and run again.")