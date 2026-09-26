import librosa
import numpy as np

CLIPPING_LEVEL = 0.99
CLIPPING_RATIO_BAD = 0.01
SILENCE_RMS = 0.001
LOW_SIGNAL_RMS = 0.01
SHORT_DURATION_SEC = 0.5


def flatness_of_loud_frames(samples):
    flatness = librosa.feature.spectral_flatness(y=samples)[0]
    levels = librosa.feature.rms(y=samples)[0]

    if len(levels) == 0 or len(flatness) == 0:
        return 1.0

    peak = float(np.max(levels))

    if peak < 1e-6:
        return 1.0

    loud = levels >= 0.15 * peak

    if not loud.any():
        return 1.0

    return float(np.mean(flatness[: len(loud)][loud[: len(flatness)]]))


def steadiness(samples, sr):
    frame = max(int(0.02 * sr), 1)
    frame_count = len(samples) // frame

    if frame_count < 3:
        return 0.0

    levels = []
    for i in range(frame_count):
        chunk = samples[i * frame:(i + 1) * frame]
        levels.append(float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2))))

    levels.sort()
    floor = float(np.mean(levels[: max(len(levels) // 10, 1)]))
    peak = float(np.mean(levels[-max(len(levels) // 10, 1):]))

    if peak < 1e-9:
        return 1.0

    return min(floor / peak, 1.0)


def estimate_noise_level(samples, sr):
    if len(samples) == 0:
        return 1.0

    try:
        flatness = flatness_of_loud_frames(samples)
    except Exception:
        flatness = 1.0

    return round(min(flatness, steadiness(samples, sr)), 4)


def detect_silence(samples):
    if len(samples) == 0:
        return True, 0.0

    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    return rms < SILENCE_RMS, round(rms, 6)


def detect_clipping(samples):
    if len(samples) == 0:
        return False, 0.0

    clipped = int(np.sum(np.abs(samples) >= CLIPPING_LEVEL))
    ratio = clipped / len(samples)
    return ratio > CLIPPING_RATIO_BAD, round(ratio, 5)


def analyse_quality(samples, sr):
    samples = np.asarray(samples, dtype=np.float32)

    if samples.ndim > 1:
        samples = np.mean(samples, axis=0)

    samples = np.nan_to_num(samples)

    notes = []
    duration = len(samples) / sr if sr else 0.0

    is_silent, rms = detect_silence(samples)
    is_clipped, clip_ratio = detect_clipping(samples)
    noise_level = estimate_noise_level(samples, sr)

    if samples.size == 0:
        return {
            "quality": "Unusable",
            "notes": "The recording contains no audio frames.",
            "noise_level": 1.0,
            "rms": 0.0,
            "clipping_ratio": 0.0,
            "duration_sec": 0.0,
            "is_silent": True,
            "is_clipped": False,
        }

    if is_silent:
        notes.append("The recording is silent or almost silent.")

    if is_clipped:
        notes.append(f"Severe clipping detected in {clip_ratio * 100:.1f}% of samples.")

    if not is_silent and rms < LOW_SIGNAL_RMS:
        notes.append("Signal strength is low.")

    if noise_level > 0.6:
        notes.append("Background noise is high compared to the event.")
    elif noise_level > 0.35:
        notes.append("Some background noise is present.")

    if duration < SHORT_DURATION_SEC:
        notes.append(f"The recording is very short ({duration:.2f}s).")

    if is_silent or samples.size == 0:
        quality = "Unusable"
    elif is_clipped or noise_level > 0.6 or rms < LOW_SIGNAL_RMS or duration < SHORT_DURATION_SEC:
        quality = "Poor"
    elif noise_level > 0.35:
        quality = "Acceptable"
    else:
        quality = "Good"

    if not notes:
        notes.append("No audio quality problems found.")

    return {
        "quality": quality,
        "notes": " ".join(notes),
        "noise_level": noise_level,
        "rms": rms,
        "clipping_ratio": clip_ratio,
        "duration_sec": round(duration, 3),
        "is_silent": is_silent,
        "is_clipped": is_clipped,
    }
