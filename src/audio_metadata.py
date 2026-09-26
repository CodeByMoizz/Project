import hashlib
import os
import sys
from datetime import datetime

import numpy as np
import soundfile as sf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audio_preprocessing.preprocessing import convert_to_mono, load_audio


def file_hash(file_path):
    digest = hashlib.sha256()

    with open(file_path, "rb") as audio_file:
        for chunk in iter(lambda: audio_file.read(65536), b""):
            digest.update(chunk)

    return digest.hexdigest()


def audio_fingerprint(samples, sr):
    samples = convert_to_mono(samples)
    samples = np.nan_to_num(samples)

    if samples.size == 0:
        return None

    peak = np.max(np.abs(samples))
    if peak > 0:
        samples = samples / peak

    buckets = 32
    if len(samples) < buckets:
        samples = np.pad(samples, (0, buckets - len(samples)))

    chunks = np.array_split(samples, buckets)
    levels = np.array([float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2))) for chunk in chunks])

    average = float(np.mean(levels))
    bits = "".join("1" if level >= average else "0" for level in levels)

    return bits


def bit_depth_for(file_path):
    try:
        info = sf.info(file_path)
    except Exception:
        return None

    subtype = str(info.subtype or "")

    for depth in ("8", "16", "24", "32", "64"):
        if depth in subtype:
            return int(depth)

    return None


def extract_metadata(file_path, original_filename=None):
    samples, sr = load_audio(file_path)

    channels = 1 if samples.ndim == 1 else samples.shape[0]
    frames = samples.shape[-1] if samples.ndim > 1 else len(samples)
    duration = frames / sr if sr else 0.0

    stats = os.stat(file_path)

    return {
        "filename": original_filename or os.path.basename(file_path),
        "audio_format": os.path.splitext(file_path)[1].lower().lstrip("."),
        "duration_sec": round(duration, 3),
        "sample_rate": int(sr),
        "channels": int(channels),
        "bit_depth": bit_depth_for(file_path),
        "file_size": stats.st_size,
        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_hash": file_hash(file_path),
        "fingerprint": audio_fingerprint(samples, sr),
    }
