import librosa
import numpy as np
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field

# formats we allow
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# these numbers are not final, just starting values
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024
MIN_DURATION_SECONDS = 0.3
MAX_DURATION_SECONDS = 30.0
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 48000
SILENCE_RMS_THRESHOLD = 0.001


class ValidationStatus(Enum):
    VALID = "VALID"
    WARNING = "WARNING"
    INVALID = "INVALID"


@dataclass
class ValidationResult:
    check_name: str
    status: ValidationStatus
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class AudioValidationReport:
    file_path: str
    results: list = field(default_factory=list)

    @property
    def is_valid(self):
        return not any(r.status == ValidationStatus.INVALID for r in self.results)

    @property
    def has_warnings(self):
        return any(r.status == ValidationStatus.WARNING for r in self.results)

    def add(self, result):
        self.results.append(result)

    def failed_checks(self):
        return [r for r in self.results if r.status == ValidationStatus.INVALID]

    def summary(self):
        if self.is_valid and not self.has_warnings:
            return f"VALID: {self.file_path}"
        if self.is_valid and self.has_warnings:
            return f"VALID WITH WARNINGS: {self.file_path}"
        reasons = "; ".join(r.message for r in self.failed_checks())
        return f"INVALID: {self.file_path} ({reasons})"


def validate_file_exists(file_path):
    path = Path(file_path)
    if not path.exists():
        return ValidationResult("file_exists", ValidationStatus.INVALID, f"File does not exist: {file_path}")
    if not path.is_file():
        return ValidationResult("file_exists", ValidationStatus.INVALID, f"Path is not a file: {file_path}")
    return ValidationResult("file_exists", ValidationStatus.VALID, "File exists.")


def validate_extension(file_path):
    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return ValidationResult("extension", ValidationStatus.INVALID, f"Unsupported extension '{ext}'.", {"extension": ext})
    return ValidationResult("extension", ValidationStatus.VALID, f"Extension '{ext}' is supported.", {"extension": ext})


def validate_file_size(file_path):
    size_bytes = Path(file_path).stat().st_size

    if size_bytes == 0:
        return ValidationResult("file_size", ValidationStatus.INVALID, "File is empty (0 bytes).", {"size_bytes": size_bytes})

    if size_bytes > MAX_FILE_SIZE_BYTES:
        return ValidationResult("file_size", ValidationStatus.INVALID, f"File too large ({size_bytes / 1_000_000:.2f} MB).", {"size_bytes": size_bytes})

    return ValidationResult("file_size", ValidationStatus.VALID, f"File size OK ({size_bytes / 1_000:.1f} KB).", {"size_bytes": size_bytes})


def validate_audio_decoding(file_path):
    # sr=None keeps original sample rate, mono=False keeps original channels
    try:
        y, sr = librosa.load(file_path, sr=None, mono=False)
    except Exception as error:
        result = ValidationResult("audio_decoding", ValidationStatus.INVALID, f"Could not decode audio: {error}")
        return result, None, None

    result = ValidationResult("audio_decoding", ValidationStatus.VALID, "File decoded successfully.")
    return result, y, sr


def validate_duration(y, sr):
    duration = librosa.get_duration(y=y, sr=sr)

    if duration < MIN_DURATION_SECONDS:
        return ValidationResult("duration", ValidationStatus.INVALID, f"Audio too short ({duration:.3f}s).", {"duration_seconds": duration})

    if duration > MAX_DURATION_SECONDS:
        return ValidationResult("duration", ValidationStatus.INVALID, f"Audio too long ({duration:.2f}s).", {"duration_seconds": duration})

    return ValidationResult("duration", ValidationStatus.VALID, f"Duration OK ({duration:.2f}s).", {"duration_seconds": duration})


def validate_sample_rate(sr):
    if sr < MIN_SAMPLE_RATE or sr > MAX_SAMPLE_RATE:
        return ValidationResult("sample_rate", ValidationStatus.WARNING, f"Unusual sample rate ({sr} Hz).", {"sample_rate": sr})
    return ValidationResult("sample_rate", ValidationStatus.VALID, f"Sample rate OK ({sr} Hz).", {"sample_rate": sr})


def validate_channels(y):
    channels = 1 if y.ndim == 1 else y.shape[0]

    if channels not in (1, 2):
        return ValidationResult("channels", ValidationStatus.INVALID, f"Unsupported channel count ({channels}).", {"channels": channels})

    label = "mono" if channels == 1 else "stereo"
    return ValidationResult("channels", ValidationStatus.VALID, f"Channel count OK ({label}).", {"channels": channels})


def validate_audio_signal(y):
    if y.size == 0:
        return ValidationResult("audio_signal", ValidationStatus.INVALID, "Audio has zero samples.")

    mono_y = y if y.ndim == 1 else y.mean(axis=0)
    rms = float(np.sqrt(np.mean(mono_y.astype(np.float64) ** 2)))

    if rms < SILENCE_RMS_THRESHOLD:
        return ValidationResult("audio_signal", ValidationStatus.INVALID, f"Audio seems silent (RMS={rms:.6f}).", {"rms": rms})

    return ValidationResult("audio_signal", ValidationStatus.VALID, f"Audio has signal (RMS={rms:.4f}).", {"rms": rms})


def validate_audio_file(file_path):
    report = AudioValidationReport(file_path=str(file_path))

    exists_result = validate_file_exists(file_path)
    report.add(exists_result)
    if exists_result.status == ValidationStatus.INVALID:
        return report

    report.add(validate_extension(file_path))
    report.add(validate_file_size(file_path))

    decode_result, y, sr = validate_audio_decoding(file_path)
    report.add(decode_result)
    if decode_result.status == ValidationStatus.INVALID or y is None:
        return report

    report.add(validate_duration(y, sr))
    report.add(validate_sample_rate(sr))
    report.add(validate_channels(y))
    report.add(validate_audio_signal(y))

    return report


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python validator.py <path_to_audio_file>")
        sys.exit(1)

    report = validate_audio_file(sys.argv[1])
    
    
    

    print(report.summary())
    print("-" * 50)
    for r in report.results:
        print(f"[{r.status.value:8}] {r.check_name:15} {r.message}")