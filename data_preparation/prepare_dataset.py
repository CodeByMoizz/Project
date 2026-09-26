import csv
import os
import random
import re
import shutil
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audio_preprocessing.validation import validate_audio_file
from config.config import (
    CLASSES,
    DATASET_DIR,
    METADATA_DIR,
    SPEECH_CLASSES,
    SUPPORTED_EXTENSIONS,
)

RAW_DIR = DATASET_DIR / "raw"
SPLIT_DIRS = {
    "training": DATASET_DIR / "training",
    "validation": DATASET_DIR / "validation",
    "testing": DATASET_DIR / "testing",
}

METADATA_FILE = METADATA_DIR / "dataset_metadata.csv"
STATS_FILE = METADATA_DIR / "dataset_statistics.csv"

TARGET_PER_CLASS = 300
SPLIT_SHARES = {"training": 0.70, "validation": 0.15, "testing": 0.15}
SEED = 42

SOURCE_METADATA_FILE = METADATA_DIR / "source_metadata.csv"

METADATA_COLUMNS = [
    "audio_id",
    "filename",
    "class_label",
    "source",
    "duration_sec",
    "sample_rate",
    "channels",
    "recording_environment",
    "recording_device",
    "source_distance",
    "speaker_id",
    "recording_id",
    "original_or_augmented",
    "dataset_split",
]


def read_source_metadata():
    if not os.path.exists(SOURCE_METADATA_FILE):
        return {}

    rows = {}

    with open(SOURCE_METADATA_FILE, newline="") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("filename") or "").strip()

            if name:
                rows[name] = row

    return rows


def speaker_for(filename, class_label, source_row):
    if source_row and (source_row.get("speaker_id") or "").strip():
        return source_row["speaker_id"].strip().lower()

    if class_label not in SPEECH_CLASSES:
        return ""

    stem = os.path.splitext(filename)[0]

    match = re.search(r"(?:speaker|spk)[_-]?([A-Za-z0-9]+)", stem, re.IGNORECASE)

    if match:
        return match.group(1).lower()

    first_word = stem.split("_")[0].lower()

    if not first_word or not any(character.isalpha() for character in first_word):
        return ""

    if first_word in class_label.lower().split("_"):
        return ""

    return first_word


def group_key(filename, class_label, source_row):
    speaker = speaker_for(filename, class_label, source_row)

    if speaker:
        return "speaker:" + speaker

    if source_row and (source_row.get("recording_id") or "").strip():
        return "recording:" + source_row["recording_id"].strip().lower()

    stem = os.path.splitext(filename)[0]

    recording = re.split(r"__aug|_seg\d+", stem)[0]
    return "recording:" + recording.lower()


def collect_class_files(class_label):
    class_dir = RAW_DIR / class_label

    if not class_dir.is_dir():
        return []

    files = []

    for name in sorted(os.listdir(class_dir)):
        if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS:
            files.append(name)

    return files


def split_groups(groups, shares):
    names = sorted(groups.keys())
    random.Random(SEED).shuffle(names)

    total_files = sum(len(groups[name]) for name in names)
    wanted = {split: share * total_files for split, share in shares.items()}

    assigned = {split: [] for split in shares}
    counts = {split: 0 for split in shares}

    for name in names:
        size = len(groups[name])
        target = max(wanted, key=lambda split: wanted[split] - counts[split])
        assigned[target].append(name)
        counts[target] += size

    return assigned, counts


def prepare():
    if not RAW_DIR.is_dir():
        print(f"No raw audio found. Create {RAW_DIR} with one folder per class:")
        for class_label in CLASSES:
            print(f"  {RAW_DIR / class_label}")
        return

    source_metadata = read_source_metadata()

    for split_dir in SPLIT_DIRS.values():
        if split_dir.exists():
            print(f"clearing {split_dir}")
            shutil.rmtree(split_dir)

    for split_dir in SPLIT_DIRS.values():
        for class_label in CLASSES:
            os.makedirs(split_dir / class_label, exist_ok=True)

    os.makedirs(METADATA_DIR, exist_ok=True)

    metadata_rows = []
    stats_rows = []
    warnings = []
    audio_number = 0

    for class_label in CLASSES:
        files = collect_class_files(class_label)

        if not files:
            warnings.append(f"{class_label}: no audio files found.")
            stats_rows.append(
                {"class_label": class_label, "total": 0, "training": 0, "validation": 0, "testing": 0}
            )
            continue

        groups = {}

        for name in files:
            key = group_key(name, class_label, source_metadata.get(name))
            groups.setdefault(key, []).append(name)

        assigned, counts = split_groups(groups, SPLIT_SHARES)

        for split_name, group_names in assigned.items():
            for key in group_names:
                for name in groups[key]:
                    source_path = RAW_DIR / class_label / name

                    report = validate_audio_file(source_path)

                    if not report.is_valid:
                        warnings.append(f"{class_label}/{name}: {report.summary()}")
                        continue

                    details = {}
                    for result in report.results:
                        details.update(result.details)

                    audio_number += 1
                    audio_id = f"{class_label}_{audio_number:05d}"

                    destination = SPLIT_DIRS[split_name] / class_label / name
                    shutil.copy2(source_path, destination)

                    source_row = source_metadata.get(name, {})

                    metadata_rows.append(
                        {
                            "audio_id": audio_id,
                            "filename": name,
                            "class_label": class_label,
                            "source": source_row.get("source", "collected"),
                            "duration_sec": round(details.get("duration_seconds", 0.0), 3),
                            "sample_rate": details.get("sample_rate", ""),
                            "channels": details.get("channels", ""),
                            "recording_environment": source_row.get("recording_environment", ""),
                            "recording_device": source_row.get("recording_device", ""),
                            "source_distance": source_row.get("source_distance", ""),
                            "speaker_id": speaker_for(name, class_label, source_row),
                            "recording_id": key.split(":", 1)[1],
                            # Augmented clips are produced later by augmentation/augmentation.py and are never counted as unique original recordings here.
                            "original_or_augmented": "original",
                            "dataset_split": split_name,
                        }
                    )

        class_total = sum(
            1 for row in metadata_rows if row["class_label"] == class_label
        )

        stats_rows.append(
            {
                "class_label": class_label,
                "total": class_total,
                "training": counts["training"],
                "validation": counts["validation"],
                "testing": counts["testing"],
            }
        )

        if class_total < TARGET_PER_CLASS:
            warnings.append(
                f"{class_label}: only {class_total} clips, {TARGET_PER_CLASS} are required."
            )

    with open(METADATA_FILE, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerows(metadata_rows)

    with open(STATS_FILE, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["class_label", "total", "training", "validation", "testing"]
        )
        writer.writeheader()
        writer.writerows(stats_rows)

    check_balance(stats_rows, warnings)
    check_leakage(metadata_rows, warnings)

    print(f"\nTotal original clips: {len(metadata_rows)}")
    print(f"Metadata written to {METADATA_FILE}")
    print(f"Statistics written to {STATS_FILE}")

    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for warning in warnings:
            print("  -", warning)
    else:
        print("\nNo dataset problems found.")


def check_balance(stats_rows, warnings):
    totals = [row["total"] for row in stats_rows if row["total"] > 0]

    if not totals:
        return

    print("\nClass balance:")
    for row in stats_rows:
        print(
            f"  {row['class_label']:22} total={row['total']:5} "
            f"train={row['training']:5} val={row['validation']:5} test={row['testing']:5}"
        )

    smallest = min(totals)
    largest = max(totals)

    if smallest and largest / smallest > 1.5:
        warnings.append(
            f"The dataset is imbalanced: the largest class has {largest} clips, "
            f"the smallest has {smallest}."
        )


def check_leakage(metadata_rows, warnings):
    recording_splits = {}
    speaker_splits = {}

    for row in metadata_rows:
        recording_splits.setdefault(row["recording_id"], set()).add(row["dataset_split"])

        if row["speaker_id"]:
            speaker_splits.setdefault(row["speaker_id"], set()).add(row["dataset_split"])

    for recording, splits in recording_splits.items():
        if len(splits) > 1:
            warnings.append(
                f"Recording '{recording}' appears in several splits: {', '.join(sorted(splits))}."
            )

    for speaker, splits in speaker_splits.items():
        if len(splits) > 1:
            warnings.append(
                f"Speaker '{speaker}' appears in several splits: {', '.join(sorted(splits))}."
            )


if __name__ == "__main__":
    prepare()
