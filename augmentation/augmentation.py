from pathlib import Path
import random

import numpy as np
import pandas as pd
import librosa
import soundfile as sf

from scipy.signal import fftconvolve


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "train"
OUTPUT_DIR = PROJECT_ROOT / "data" / "augmented" / "train"
METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "augmentation_metadata.csv"
)

SAMPLE_RATE = 16000
AUGMENTATIONS_PER_FILE = 3
SEED = 42


random.seed(SEED)
np.random.seed(SEED)


def load_audio(file_path, sr=SAMPLE_RATE):
    audio, _ = librosa.load(
        file_path,
        sr=sr,
        mono=True
    )

    return audio.astype(np.float32)


def safe_normalize(audio, peak=0.98):
    max_value = np.max(np.abs(audio))

    if max_value == 0:
        return audio

    audio = audio / max_value
    audio = audio * peak

    return audio.astype(np.float32)


def add_noise(audio, noise_factor=None):
    if noise_factor is None:
        noise_factor = random.uniform(
            0.002,
            0.015
        )

    noise = np.random.normal(
        0,
        noise_factor,
        size=len(audio)
    ).astype(np.float32)

    augmented = audio + noise

    return safe_normalize(augmented)


def time_shift(audio, shift_max_seconds=0.25):
    shift_max = int(
        shift_max_seconds * SAMPLE_RATE
    )

    shift = random.randint(
        -shift_max,
        shift_max
    )

    if shift == 0:
        return audio.copy()

    augmented = np.roll(
        audio,
        shift
    )

    if shift > 0:
        augmented[:shift] = 0
    else:
        augmented[shift:] = 0

    return augmented.astype(np.float32)


def pitch_shift(audio):
    steps = random.uniform(
        -1.5,
        1.5
    )

    augmented = librosa.effects.pitch_shift(
        y=audio,
        sr=SAMPLE_RATE,
        n_steps=steps
    )

    return safe_normalize(augmented)


def time_stretch(audio):
    rate = random.uniform(
        0.90,
        1.10
    )

    augmented = librosa.effects.time_stretch(
        y=audio,
        rate=rate
    )

    target_length = len(audio)

    if len(augmented) > target_length:
        augmented = augmented[:target_length]

    elif len(augmented) < target_length:
        augmented = np.pad(
            augmented,
            (
                0,
                target_length - len(augmented)
            )
        )

    return safe_normalize(augmented)


def volume_adjustment(audio):
    gain_db = random.uniform(
        -6,
        6
    )

    gain = 10 ** (
        gain_db / 20
    )

    augmented = audio * gain

    return safe_normalize(augmented)


def add_reverb(audio):
    duration = random.uniform(
        0.15,
        0.35
    )

    reverb_length = int(
        duration * SAMPLE_RATE
    )

    impulse = np.zeros(
        reverb_length
    )

    impulse[0] = 1.0

    decay = random.uniform(
        0.3,
        0.6
    )

    for i in range(
        1,
        reverb_length
    ):
        impulse[i] = (
            decay ** (
                i / SAMPLE_RATE * 10
            )
        ) * random.uniform(
            0.7,
            1.0
        )

    reverberated = fftconvolve(
        audio,
        impulse,
        mode="full"
    )

    reverberated = reverberated[
        :len(audio)
    ]

    wet = random.uniform(
        0.10,
        0.30
    )

    augmented = (
        (1 - wet) * audio
        + wet * reverberated
    )

    return safe_normalize(augmented)


def simulate_distance(audio):
    gain = random.uniform(
        0.35,
        0.75
    )

    distant = audio * gain

    noise = np.random.normal(
        0,
        random.uniform(
            0.001,
            0.005
        ),
        len(distant)
    )

    distant = distant + noise

    return safe_normalize(distant)


def simulate_recording_device(audio):
    gain_db = random.uniform(
        -4,
        4
    )

    gain = 10 ** (
        gain_db / 20
    )

    augmented = audio * gain

    noise_level = random.uniform(
        0.0005,
        0.003
    )

    noise = np.random.normal(
        0,
        noise_level,
        len(augmented)
    )

    augmented = augmented + noise

    return safe_normalize(augmented)


AUGMENTATION_FUNCTIONS = {
    "noise": add_noise,
    "time_shift": time_shift,
    "pitch_shift": pitch_shift,
    "time_stretch": time_stretch,
    "volume": volume_adjustment,
    "reverb": add_reverb,
    "distance": simulate_distance,
    "device": simulate_recording_device,
}


def apply_random_augmentation(audio):
    number_of_operations = random.randint(
        1,
        3
    )

    operations = random.sample(
        list(AUGMENTATION_FUNCTIONS.keys()),
        number_of_operations
    )

    augmented = audio.copy()

    for operation in operations:
        function = AUGMENTATION_FUNCTIONS[
            operation
        ]

        augmented = function(
            augmented
        )

    return augmented, operations


def save_audio(audio, output_path):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sf.write(
        output_path,
        audio,
        SAMPLE_RATE,
        subtype="PCM_16"
    )


def process_class(class_name, metadata_rows):
    input_class_dir = (
        INPUT_DIR / class_name
    )

    output_class_dir = (
        OUTPUT_DIR / class_name
    )

    output_class_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    audio_files = list(
        input_class_dir.glob("*.wav")
    )

    print(
        f"\nClass: {class_name}"
    )

    print(
        f"Original training files: "
        f"{len(audio_files)}"
    )

    for file_path in audio_files:
        try:
            audio = load_audio(
                file_path
            )

            for index in range(
                AUGMENTATIONS_PER_FILE
            ):
                augmented_audio, operations = (
                    apply_random_augmentation(
                        audio
                    )
                )

                output_name = (
                    f"{file_path.stem}"
                    f"__aug_{index + 1}.wav"
                )

                output_path = (
                    output_class_dir
                    / output_name
                )

                save_audio(
                    augmented_audio,
                    output_path
                )

                metadata_rows.append(
                    {
                        "audio_id":
                            f"{file_path.stem}"
                            f"__aug_{index + 1}",

                        "original_audio_id":
                            file_path.stem,

                        "filename":
                            output_name,

                        "class_label":
                            class_name,

                        "original_or_augmented":
                            "augmented",

                        "dataset_split":
                            "training",

                        "augmentation_operations":
                            "|".join(
                                operations
                            ),

                        "source_file":
                            file_path.name,

                        "sample_rate":
                            SAMPLE_RATE
                    }
                )

        except Exception as e:
            print(
                f"ERROR processing "
                f"{file_path}: {e}"
            )


def main():
    print("=" * 60)

    print(
        "SonicSentinel AI - Audio Augmentation"
    )

    print("=" * 60)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Training directory not found: "
            f"{INPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    METADATA_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata_rows = []

    class_directories = [
        directory
        for directory in INPUT_DIR.iterdir()
        if directory.is_dir()
    ]

    if not class_directories:
        raise ValueError(
            "No class directories found."
        )

    for class_directory in sorted(
        class_directories
    ):
        process_class(
            class_directory.name,
            metadata_rows
        )

    metadata_df = pd.DataFrame(
        metadata_rows
    )

    metadata_df.to_csv(
        METADATA_FILE,
        index=False
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "Augmentation completed successfully."
    )

    print(
        f"Generated augmented files: "
        f"{len(metadata_df)}"
    )

    print(
        f"Metadata saved to: "
        f"{METADATA_FILE}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()