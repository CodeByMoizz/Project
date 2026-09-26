import os
import random
import shutil
import sys

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import fftconvolve, lfilter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import (
    DATASET_DIR,
    METADATA_DIR,
    SUPPORTED_EXTENSIONS,
    TARGET_SAMPLE_RATE,
)

# Augmentation only ever runs on the training split, and the output stays in the
# training split. Augmented clips are never counted as original recordings.
INPUT_DIR = DATASET_DIR / "training"
OUTPUT_DIR = DATASET_DIR / "augmented_training"
METADATA_FILE = METADATA_DIR / "augmentation_metadata.csv"

SAMPLE_RATE = TARGET_SAMPLE_RATE
AUGMENTATIONS_PER_FILE = 3
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def load_audio(file_path, sr=SAMPLE_RATE):
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    return audio.astype(np.float32)


def safe_normalize(audio, peak=0.98):
    max_value = np.max(np.abs(audio))

    if max_value == 0:
        return audio.astype(np.float32)

    return (audio / max_value * peak).astype(np.float32)


def add_noise(audio):
    noise_factor = random.uniform(0.002, 0.015)
    noise = np.random.normal(0, noise_factor, size=len(audio)).astype(np.float32)
    return safe_normalize(audio + noise)


def time_shift(audio, shift_max_seconds=0.25):
    shift_max = int(shift_max_seconds * SAMPLE_RATE)
    shift = random.randint(-shift_max, shift_max)

    if shift == 0:
        return audio.copy()

    shifted = np.roll(audio, shift)

    if shift > 0:
        shifted[:shift] = 0
    else:
        shifted[shift:] = 0

    return shifted.astype(np.float32)


def pitch_shift(audio):
    steps = random.uniform(-1.5, 1.5)
    shifted = librosa.effects.pitch_shift(y=audio, sr=SAMPLE_RATE, n_steps=steps)
    return safe_normalize(shifted)


def time_stretch(audio):
    rate = random.uniform(0.90, 1.10)
    stretched = librosa.effects.time_stretch(y=audio, rate=rate)
    return safe_normalize(fit_length(stretched, len(audio)))


def fit_length(audio, target_length):
    if len(audio) > target_length:
        return audio[:target_length]

    if len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)))

    return audio


def volume_adjustment(audio):
    gain_db = random.uniform(-6, 6)
    gain = 10 ** (gain_db / 20)
    return safe_normalize(audio * gain)


def add_reverb(audio):
    duration = random.uniform(0.15, 0.35)
    reverb_length = int(duration * SAMPLE_RATE)

    decay = random.uniform(0.3, 0.6)
    positions = np.arange(reverb_length)

    impulse = (decay ** (positions / SAMPLE_RATE * 10)) * np.random.uniform(
        0.7, 1.0, size=reverb_length
    )
    impulse[0] = 1.0

    reverberated = fftconvolve(audio, impulse, mode="full")[: len(audio)]

    wet = random.uniform(0.10, 0.30)
    return safe_normalize((1 - wet) * audio + wet * reverberated)


# A distant sound is quieter, duller and more reverberant than a close one.
def distance_simulation(audio):
    attenuation = random.uniform(0.25, 0.65)
    cutoff = random.uniform(2500, 6000)

    quieter = audio * attenuation
    dulled = low_pass(quieter, cutoff)

    wet = random.uniform(0.15, 0.35)
    reverberated = add_reverb(dulled)

    return safe_normalize((1 - wet) * dulled + wet * reverberated)


# Different microphones colour the sound, so the frequency response is tilted
# and a little noise floor is added.
def device_simulation(audio):
    if random.random() < 0.5:
        shaped = low_pass(audio, random.uniform(4000, 7000))
    else:
        shaped = high_pass(audio, random.uniform(80, 300))

    noise_floor = np.random.normal(0, random.uniform(0.001, 0.004), size=len(shaped))
    return safe_normalize(shaped + noise_floor.astype(np.float32))


def low_pass(audio, cutoff):
    alpha = cutoff / (cutoff + SAMPLE_RATE / (2 * np.pi))
    return lfilter([alpha], [1, -(1 - alpha)], audio).astype(np.float32)


def high_pass(audio, cutoff):
    return (audio - low_pass(audio, cutoff)).astype(np.float32)


# The eight augmentation types the SRS lists.
AUGMENTATION_FUNCTIONS = {
    "noise": add_noise,
    "time_shift": time_shift,
    "pitch_shift": pitch_shift,
    "time_stretch": time_stretch,
    "volume": volume_adjustment,
    "reverb": add_reverb,
    "distance": distance_simulation,
    "device": device_simulation,
}


def apply_random_augmentation(audio):
    number_of_operations = random.randint(1, 3)
    operations = random.sample(list(AUGMENTATION_FUNCTIONS.keys()), number_of_operations)

    augmented = audio.copy()

    for operation in operations:
        augmented = AUGMENTATION_FUNCTIONS[operation](augmented)

    return augmented, operations


def save_audio(audio, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, SAMPLE_RATE, subtype="PCM_16")


def process_class(class_name, metadata_rows):
    input_class_dir = INPUT_DIR / class_name
    output_class_dir = OUTPUT_DIR / class_name
    output_class_dir.mkdir(parents=True, exist_ok=True)

    # Every supported format, not just wav, because the dataset also contains
    # mp3 originals. The augmented copies are always written as wav.
    audio_files = sorted(
        f for f in input_class_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    print(f"\nClass: {class_name}")
    print(f"Original training files: {len(audio_files)}")

    for file_path in audio_files:
        try:
            audio = load_audio(file_path)
        except Exception as error:
            print(f"ERROR loading {file_path}: {error}")
            continue

        for index in range(AUGMENTATIONS_PER_FILE):
            try:
                augmented_audio, operations = apply_random_augmentation(audio)
            except Exception as error:
                print(f"ERROR augmenting {file_path}: {error}")
                continue

            output_name = f"{file_path.stem}__aug_{index + 1}.wav"
            save_audio(augmented_audio, output_class_dir / output_name)

            metadata_rows.append(
                {
                    "audio_id": f"{file_path.stem}__aug_{index + 1}",
                    "original_audio_id": file_path.stem,
                    "filename": output_name,
                    "class_label": class_name,
                    "original_or_augmented": "augmented",
                    "dataset_split": "training",
                    "augmentation_operations": "|".join(operations),
                    "source_file": file_path.name,
                    "sample_rate": SAMPLE_RATE,
                }
            )


def main():
    print("=" * 60)
    print("SonicSentinel AI - Audio Augmentation (training split only)")
    print("=" * 60)

    if not INPUT_DIR.exists():
        print(f"Training split not found: {INPUT_DIR}")
        print("Run data_preparation/prepare_dataset.py first.")
        return

    # Rebuilt from the training split every time, for the same reason the split
    # folders are cleared in prepare_dataset.py.
    if OUTPUT_DIR.exists():
        print(f"clearing {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    class_directories = sorted(d for d in INPUT_DIR.iterdir() if d.is_dir())

    if not class_directories:
        print("No class directories found in the training split.")
        return

    metadata_rows = []

    for class_directory in class_directories:
        process_class(class_directory.name, metadata_rows)

    pd.DataFrame(metadata_rows).to_csv(METADATA_FILE, index=False)

    print("\n" + "=" * 60)
    print(f"Generated augmented files: {len(metadata_rows)}")
    print(f"Metadata saved to: {METADATA_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
