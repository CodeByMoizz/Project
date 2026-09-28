import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Change this one line to switch the active Python model after training.
# It must match a key in PYTHON_MODELS below.
#
# "kind" says what the model reads. "features" means the 215 summary values from
# feature_extraction/features.py. "spectrogram" means the log-mel image from
# feature_extraction/spectrogram.py, which is what the CNN uses.
ACTIVE_PYTHON_MODEL = "xgboost"

PYTHON_MODELS = {
    "random_forest": {
        "kind": "features",
        "model_file": "random_forest_model.pkl",
        "scaler_file": "random_forest_scaler.pkl",
        "metrics_file": "random_forest_metrics.json",
        "version": "rf-v1",
    },
    "svm": {
        "kind": "features",
        "model_file": "svm_model.pkl",
        "scaler_file": "svm_scaler.pkl",
        "metrics_file": "svm_metrics.json",
        "version": "svm-v1",
    },
    "xgboost": {
        "kind": "features",
        "model_file": "xgboost_model.pkl",
        "scaler_file": "xgboost_scaler.pkl",
        "metrics_file": "xgboost_metrics.json",
        "version": "xgb-v1",
    },
    "logistic_regression": {
        "kind": "features",
        "model_file": "logistic_regression_model.pkl",
        "scaler_file": "logistic_regression_scaler.pkl",
        "metrics_file": "logistic_regression_metrics.json",
        "version": "lr-v1",
    },
    # The CNN reads a log-mel spectrogram, so it needs no scaler: the
    # spectrogram is scaled to a fixed decibel range instead.
    "cnn": {
        "kind": "spectrogram",
        "model_file": "cnn_model.keras",
        "scaler_file": None,
        "metrics_file": "cnn_metrics.json",
        "version": "cnn-v1",
    },
    # Transfer learning: a small head on frozen YAMNet embeddings.
    "yamnet": {
        "kind": "embedding",
        "model_file": "yamnet_model.keras",
        "scaler_file": None,
        "metrics_file": "yamnet_metrics.json",
        "version": "yamnet-v1",
    },
    # The same frozen YAMNet embeddings, but with a scikit-learn head chosen on
    # the validation split by src/yamnet_transfer.py. The pipeline carries its
    # own scaler, so there is no separate scaler file.
    "yamnet_transfer": {
        "kind": "embedding_sklearn",
        "model_file": "yamnet_transfer.joblib",
        "scaler_file": None,
        "metrics_file": "yamnet_transfer_metrics.json",
        "version": "yamnet-transfer-v1",
    },
}

PYTHON_MODEL_DIR = PROJECT_ROOT / "python_models"
LABEL_ENCODER_FILE = "label_encoder.pkl"

# The GTM model is exported from Teachable Machine as TensorFlow.js and runs in
# the browser. Put model.json, metadata.json and the .bin weights in gtm_model/,
# or set GTM_MODEL_URL to the hosted Teachable Machine URL instead.
GTM_MODEL_DIR = PROJECT_ROOT / "gtm_model"
GTM_MODEL_URL = os.environ.get("GTM_MODEL_URL", "")
GTM_MODEL_VERSION = "gtm-v1"

ALERT_RULES_FILE = PROJECT_ROOT / "alert_rules" / "alert_rules.json"

DATABASE_FILE = PROJECT_ROOT / "database" / "sonicsentinel.db"

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
SEGMENT_DIR = PROJECT_ROOT / "data" / "segments"
VISUAL_DIR = PROJECT_ROOT / "static" / "visuals"
METRICS_DIR = PROJECT_ROOT / "reports"
DATASET_DIR = PROJECT_ROOT / "audio_dataset"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

SECRET_KEY = os.environ.get("SONICSENTINEL_SECRET_KEY", "sonicsentinel-dev-key")

# Audio settings. These must stay the same for training and for inference,
# because the feature vector changes if they do.
TARGET_SAMPLE_RATE = 16000

# 3 seconds, because nearly every dataset clip is exactly 3 seconds long. At the
# old 2 seconds a third of each clip was thrown away, which cost about 3 points
# of validation accuracy.
SEGMENT_DURATION_SEC = 3.0
SILENCE_TOP_DB = 30

# Noise reduction is available in audio_preprocessing/preprocessing.py and can be
# switched back on here, but it is off by default: measured on the validation
# split it lost about 2 points of accuracy, because subtracting an estimated
# noise floor also removes part of a quiet event.
APPLY_NOISE_REDUCTION = False

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_BATCH_FILES = 20
SUPPORTED_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]

# Defaults for the admin-configurable thresholds. The live values are kept in
# the settings table so an administrator can change them without a restart.
DEFAULT_SETTINGS = {
    "min_confidence": 0.60,
    "top_two_margin": 0.15,
    "max_confidence_difference": 0.30,
    "repeated_detection_window_sec": 30,
    "noise_alert_level": 0.50,
    "overlap_confidence": 0.25,
    "retention_days": 90,
    "failed_login_alert_count": 5,
}

# Canonical class ids. Used as folder names, model labels and database values.
CLASSES = [
    "machinery_fault",
    "glass_breaking",
    "alarm_siren",
    "vehicle_horn",
    "animal_sound",
    "gunshot",
    "panic_scream",
    "aggression",
    "person_asking_help",
    "background_noise",
]

CLASS_NAMES = {
    "machinery_fault": "Machinery Fault",
    "glass_breaking": "Glass Breaking",
    "alarm_siren": "Alarm or Siren",
    "vehicle_horn": "Vehicle Horn",
    "animal_sound": "Animal Sound",
    "gunshot": "Gunshot",
    "panic_scream": "Panic Scream",
    "aggression": "Aggression",
    "person_asking_help": "Person Asking for Help",
    "background_noise": "Background Noise",
    "unknown": "Unknown Sound",
}

CRITICAL_CLASSES = [
    "gunshot",
    "panic_scream",
    "person_asking_help",
    "aggression",
    "glass_breaking",
]

# Classes whose recordings contain speech. The dataset split has to group these
# by speaker as well as by recording, otherwise recall is inflated.
SPEECH_CLASSES = [
    "panic_scream",
    "aggression",
    "person_asking_help",
]

ROLES = ["user", "reviewer", "operator", "maintenance", "admin"]

EVENT_STATUSES = [
    "Uploaded",
    "Classified",
    "Uncertain",
    "Alert Generated",
    "Manual Review",
    "Reviewed",
    "Closed",
]


def display_name(class_id):
    return CLASS_NAMES.get(class_id, class_id)


# src/yamnet_transfer.py --activate writes this file. It overrides the default
# above without editing this module, and deleting it restores the default.
SELECTION_FILE = PYTHON_MODEL_DIR / "selection.json"


def selected_model_name():
    """The model named by selection.json, or None when there is no valid
    selection and the ACTIVE_PYTHON_MODEL default should stand."""
    try:
        with open(SELECTION_FILE) as handle:
            chosen = json.load(handle).get("selected")
    except (OSError, ValueError):
        return None

    return chosen if chosen in PYTHON_MODELS else None


def active_model_config():
    return PYTHON_MODELS[selected_model_name() or ACTIVE_PYTHON_MODEL]
