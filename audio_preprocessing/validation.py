

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import soundfile as sf
import numpy as np


# ---------------------------------------------------------------------------
# 1. CONSTANTS / PROJECT RULES
# ---------------------------------------------------------------------------
# Anything in this block is a *project decision*, not a law of physics.
# Keeping them at the top of the file means you can tune them in one
# place once real data is available, instead of hunting through the code.

# Explicitly required by the SRS (Functional Requirement iv / Section 1.9.2)
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# PROVISIONAL - not defined by the SRS. A generous upper bound just to
# reject obviously wrong uploads (e.g. someone uploading a 2 GB video
# renamed to .wav). Re-tune once you know your real file sizes.
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

# PROVISIONAL - the SRS says duration must be validated, but does not
# give numbers. These are placeholders until the dataset tells us the
# real distribution of clip lengths (Section "Hint": 3,000 clips,
# 1-3 second live windows, etc.).
MIN_DURATION_SECONDS = 0.3
MAX_DURATION_SECONDS = 30.0

# PROVISIONAL - most training pipelines standardize on 16 kHz or
# 22.05 kHz for speech/event audio. The SRS never fixes a number,
# so we WARN (not reject) outside this range instead of failing hard,
# since resampling can fix this later in preprocessing.py.
EXPECTED_MIN_SAMPLE_RATE = 8000
EXPECTED_MAX_SAMPLE_RATE = 48000

# PROVISIONAL - RMS (root-mean-square) amplitude below this is treated
# as "no usable signal". Real silence is 0.0 exactly; real-world
# "silence" usually has tiny microphone noise, so we use a small
# nonzero threshold rather than checking for a literal zero.
SILENCE_RMS_THRESHOLD = 0.001


class ValidationStatus(Enum):
    """The three possible outcomes of a single check."""
    VALID = "VALID"
    WARNING = "WARNING"
    INVALID = "INVALID"


@dataclass
class ValidationResult:
    """
    The result of ONE validation check.

    check_name : short machine-friendly name, e.g. "file_exists"
    status     : VALID / WARNING / INVALID
    message    : human-readable explanation, safe to show in a UI or log
    details    : optional extra data (e.g. the actual duration found),
                 useful for debugging or for building a report later
    """
    check_name: str
    status: ValidationStatus
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class AudioValidationReport:
    """
    The combined result of running ALL checks on one file.

    is_valid       : True only if every check passed as VALID or WARNING
                      (WARNING never blocks the pipeline, INVALID always does)
    results        : the full list of individual ValidationResult objects,
                      in the order they were run
    file_path      : the file that was checked
    """
    file_path: str
    results: list = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(r.status == ValidationStatus.INVALID for r in self.results)

    @property
    def has_warnings(self) -> bool:
        return any(r.status == ValidationStatus.WARNING for r in self.results)

    def add(self, result: ValidationResult) -> None:
        self.results.append(result)

    def failed_checks(self) -> list:
        return [r for r in self.results if r.status == ValidationStatus.INVALID]

    def summary(self) -> str:
        if self.is_valid and not self.has_warnings:
            return f"VALID: {self.file_path}"
        if self.is_valid and self.has_warnings:
            return f"VALID WITH WARNINGS: {self.file_path}"
        reasons = "; ".join(r.message for r in self.failed_checks())
        return f"INVALID: {self.file_path} ({reasons})"


# ---------------------------------------------------------------------------
# 2. INDIVIDUAL CHECKS
# ---------------------------------------------------------------------------
# Each function checks exactly ONE thing and returns exactly ONE
# ValidationResult. This makes each function trivial to test on its own.


def validate_file_exists(file_path: str) -> ValidationResult:
    """Check that the path actually points to a real file on disk."""
    path = Path(file_path)
    if not path.exists():
        return ValidationResult(
            "file_exists",
            ValidationStatus.INVALID,
            f"File does not exist: {file_path}",
        )
    if not path.is_file():
        return ValidationResult(
            "file_exists",
            ValidationStatus.INVALID,
            f"Path exists but is not a file: {file_path}",
        )
    return ValidationResult(
        "file_exists", ValidationStatus.VALID, "File exists on disk."
    )


def validate_extension(file_path: str) -> ValidationResult:
    """Check that the file extension is one of the formats we support."""
    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return ValidationResult(
            "extension",
            ValidationStatus.INVALID,
            f"Unsupported file extension '{ext}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}",
            details={"extension": ext},
        )
    return ValidationResult(
        "extension",
        ValidationStatus.VALID,
        f"Extension '{ext}' is supported.",
        details={"extension": ext},
    )


def validate_file_size(file_path: str) -> ValidationResult:
    """Check that the file is neither empty nor unreasonably large."""
    size_bytes = Path(file_path).stat().st_size

    if size_bytes == 0:
        return ValidationResult(
            "file_size",
            ValidationStatus.INVALID,
            "File is empty (0 bytes).",
            details={"size_bytes": size_bytes},
        )

    if size_bytes > MAX_FILE_SIZE_BYTES:
        return ValidationResult(
            "file_size",
            ValidationStatus.INVALID,
            f"File is too large ({size_bytes / 1_000_000:.2f} MB). "
            f"Maximum allowed is {MAX_FILE_SIZE_BYTES / 1_000_000:.0f} MB "
            f"(PROVISIONAL limit).",
            details={"size_bytes": size_bytes},
        )

    return ValidationResult(
        "file_size",
        ValidationStatus.VALID,
        f"File size OK ({size_bytes / 1_000:.1f} KB).",
        details={"size_bytes": size_bytes},
    )


def validate_audio_decoding(file_path: str) -> tuple:
    """
    Attempt to open the file as audio and read its basic info.

    This is the most important check: it proves the file is not
    corrupted and is actually decodable as audio, not just a file
    that happens to have the right extension.

    Returns a tuple: (ValidationResult, soundfile.Info or None)
    The Info object (if not None) is reused by later checks so we
    don't have to re-open the file from disk multiple times.
    """
    try:
        info = sf.info(file_path)
    except Exception as error:
        return (
            ValidationResult(
                "audio_decoding",
                ValidationStatus.INVALID,
                f"File could not be decoded as audio. It may be corrupted, "
                f"empty, or mislabeled. Underlying error: {error}",
            ),
            None,
        )

    return (
        ValidationResult(
            "audio_decoding",
            ValidationStatus.VALID,
            "File was successfully opened and decoded.",
        ),
        info,
    )


def validate_duration(info) -> ValidationResult:
    """Check the clip's duration against project rules (PROVISIONAL)."""
    duration_seconds = info.frames / info.samplerate

    if duration_seconds < MIN_DURATION_SECONDS:
        return ValidationResult(
            "duration",
            ValidationStatus.INVALID,
            f"Audio is too short ({duration_seconds:.3f}s). "
            f"Minimum is {MIN_DURATION_SECONDS}s (PROVISIONAL).",
            details={"duration_seconds": duration_seconds},
        )

    if duration_seconds > MAX_DURATION_SECONDS:
        return ValidationResult(
            "duration",
            ValidationStatus.INVALID,
            f"Audio is too long ({duration_seconds:.2f}s). "
            f"Maximum is {MAX_DURATION_SECONDS}s (PROVISIONAL).",
            details={"duration_seconds": duration_seconds},
        )

    return ValidationResult(
        "duration",
        ValidationStatus.VALID,
        f"Duration OK ({duration_seconds:.2f}s).",
        details={"duration_seconds": duration_seconds},
    )


def validate_sample_rate(info) -> ValidationResult:
    """
    Check the sample rate is in a sane range.

    This is a WARNING, not an INVALID, because an unusual sample rate
    can be fixed later by resampling in preprocessing.py. It should
    not cause the whole file to be thrown away at the validation stage.
    """
    sr = info.samplerate

    if sr < EXPECTED_MIN_SAMPLE_RATE or sr > EXPECTED_MAX_SAMPLE_RATE:
        return ValidationResult(
            "sample_rate",
            ValidationStatus.WARNING,
            f"Unusual sample rate ({sr} Hz). Expected between "
            f"{EXPECTED_MIN_SAMPLE_RATE} and {EXPECTED_MAX_SAMPLE_RATE} Hz "
            f"(PROVISIONAL range). Will likely need resampling later.",
            details={"sample_rate": sr},
        )

    return ValidationResult(
        "sample_rate",
        ValidationStatus.VALID,
        f"Sample rate OK ({sr} Hz).",
        details={"sample_rate": sr},
    )


def validate_channels(info) -> ValidationResult:
    """
    Check the number of audio channels.

    Mono (1) and stereo (2) are both accepted, since preprocessing.py
    will later convert stereo to mono anyway. Anything else (0, or
    more than 2) is treated as invalid because it is not normal
    speech/event audio and likely indicates a bad or unusual file.
    """
    channels = info.channels

    if channels not in (1, 2):
        return ValidationResult(
            "channels",
            ValidationStatus.INVALID,
            f"Unsupported channel count ({channels}). Expected 1 (mono) "
            f"or 2 (stereo).",
            details={"channels": channels},
        )

    status = ValidationStatus.VALID
    message = f"Channel count OK ({'mono' if channels == 1 else 'stereo'})."
    return ValidationResult("channels", status, message, details={"channels": channels})


def validate_audio_signal(file_path: str) -> ValidationResult:
    """
    Check that the file actually contains a usable signal, i.e. it is
    not silent or filled with digital zeros.

    Unlike the previous checks, this one has to read the actual audio
    SAMPLES (not just metadata), so it is the most expensive check and
    is run last, after every cheaper check has already passed.
    """
    try:
        samples, _ = sf.read(file_path, always_2d=False)
    except Exception as error:
        return ValidationResult(
            "audio_signal",
            ValidationStatus.INVALID,
            f"Could not read audio samples for signal check: {error}",
        )

    if samples.size == 0:
        return ValidationResult(
            "audio_signal",
            ValidationStatus.INVALID,
            "Audio file contains zero samples.",
        )

    # If stereo, collapse to mono just for this loudness check.
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))

    if rms < SILENCE_RMS_THRESHOLD:
        return ValidationResult(
            "audio_signal",
            ValidationStatus.INVALID,
            f"Audio appears to be silent (RMS={rms:.6f}, threshold="
            f"{SILENCE_RMS_THRESHOLD}, PROVISIONAL).",
            details={"rms": rms},
        )

    return ValidationResult(
        "audio_signal",
        ValidationStatus.VALID,
        f"Audio contains a usable signal (RMS={rms:.4f}).",
        details={"rms": rms},
    )


# ---------------------------------------------------------------------------
# 3. ORCHESTRATOR
# ---------------------------------------------------------------------------

def validate_audio_file(file_path: str) -> AudioValidationReport:
    """
    Run every validation check on a single audio file, in order,
    and return a complete report.

    The order matters: cheap, file-system-level checks run first.
    If a check that later checks DEPEND on fails (e.g. the file
    can't be decoded at all), we stop early instead of crashing on
    checks that need data we don't have.
    """
    report = AudioValidationReport(file_path=str(file_path))

    # --- Step 1: cheap checks that don't require opening the file ---
    exists_result = validate_file_exists(file_path)
    report.add(exists_result)
    if exists_result.status == ValidationStatus.INVALID:
        return report  # No point checking anything else.

    report.add(validate_extension(file_path))
    report.add(validate_file_size(file_path))

    # --- Step 2: try to decode the file ---
    decode_result, info = validate_audio_decoding(file_path)
    report.add(decode_result)
    if decode_result.status == ValidationStatus.INVALID or info is None:
        return report  # Can't check duration/rate/channels without `info`.

    # --- Step 3: checks that rely on the decoded metadata ---
    report.add(validate_duration(info))
    report.add(validate_sample_rate(info))
    report.add(validate_channels(info))

    # --- Step 4: the most expensive check, reading actual samples ---
    report.add(validate_audio_signal(file_path))

    return report



# 4. MANUAL SMOKE TEST (run this file directly to try it on one file)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python validator.py <path_to_audio_file>")
        sys.exit(1)

    result_report = validate_audio_file(sys.argv[1])

    print(result_report.summary())
    print("-" * 60)
    for r in result_report.results:
        print(f"[{r.status.value:8}] {r.check_name:20} {r.message}")