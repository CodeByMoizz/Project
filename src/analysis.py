import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audio_preprocessing.preprocessing import prepare_segments
from audio_preprocessing.validation import validate_audio_file
from config.config import GTM_MODEL_VERSION, display_name
from database import database
from src import audio_metadata, decision, python_model, quality, rules, visuals


def count_recent_repeats(final_class, settings):
    if not final_class or final_class == "unknown":
        return 1

    window = settings.get("repeated_detection_window_sec", 30)

    try:
        rows = database.get_recent_detections_for_class(final_class, window)
    except Exception:
        return 1

    return len(rows) + 1


def store_detection(audio_id, segment_index, segment_info, result, model_version):
    values = {
        "audio_id": audio_id,
        "segment_index": segment_index,
        "start_sec": segment_info[0] if segment_info else None,
        "end_sec": segment_info[1] if segment_info else None,
        "python_class": result["python_class"],
        "python_scores": json.dumps(result["python_scores"]),
        "python_confidence": result["python_confidence"],
        "python_model_version": model_version,
        "gtm_class": result["gtm_class"],
        "gtm_scores": json.dumps(result["gtm_scores"]) if result["gtm_available"] else None,
        "gtm_confidence": result["gtm_confidence"] if result["gtm_available"] else None,
        "gtm_model_version": GTM_MODEL_VERSION if result["gtm_available"] else None,
        "agreement_status": result["agreement_status"],
        "confidence_difference": result["confidence_difference"],
        "top_two_margin": result["top_two_margin"],
        "overlapping_classes": json.dumps(result["overlapping_classes"]),
        "quality": result["quality"],
        "quality_notes": result["quality_notes"],
        "noise_level": result["noise_level"],
        "final_class": result["final_class"],
        "confidence_level": result["confidence_level"],
        "severity": result["severity"],
        "alert_status": result["alert_status"],
        "recommended_action": result["recommended_action"],
        "manual_review_required": 1 if result["manual_review_required"] else 0,
        "manual_review_reason": result["manual_review_reason"],
        "status": result["status"],
        "waveform_path": result.get("waveform_path"),
        "spectrogram_path": result.get("spectrogram_path"),
    }

    detection_id = database.add_detection(values)

    if result["alert_status"] == "Alert Generated":
        message = (
            f"{display_name(result['final_class'])} detected "
            f"({result['severity']}). {result['recommended_action']}"
        )
        database.add_alert(detection_id, result["severity"], message)

    return detection_id


def analyse_segment(samples, sr, settings, loaded_rules, gtm_scores=None, visual_name=None):
    quality_result = quality.analyse_quality(samples, sr)

    python_scores = python_model.get_prediction(samples, sr)
    model_available = python_scores is not None

    if not model_available:
        python_scores = python_model.empty_scores()

    repeat_count = 1
    result = decision.build_decision(
        python_scores, gtm_scores or {}, quality_result, repeat_count, settings, loaded_rules
    )

    repeat_count = count_recent_repeats(result["final_class"], settings)

    if repeat_count > 1:
        result = decision.build_decision(
            python_scores, gtm_scores or {}, quality_result, repeat_count, settings, loaded_rules
        )

    result["model_available"] = model_available

    if not model_available:
        status = python_model.model_status()
        result["model_message"] = status["message"]
        result["final_class"] = "unknown"
        result["final_class_name"] = display_name("unknown")
        result["status"] = "Uncertain"
        result["manual_review_required"] = True
        result["manual_review_reason"] = (
            "The Python model is not available, so the sound could not be classified."
        )

    if visual_name:
        waveform_path, spectrogram_path = visuals.make_visuals(samples, sr, visual_name)
        result["waveform_path"] = waveform_path
        result["spectrogram_path"] = spectrogram_path

    return result


def analyse_file(file_path, original_filename, user_id, source_kind="upload"):
    report = validate_audio_file(file_path)

    if not report.is_valid:
        return [], report.summary()

    settings = database.get_settings()
    loaded_rules = rules.load_rules()

    try:
        metadata = audio_metadata.extract_metadata(file_path, original_filename)
    except Exception as error:
        return [], f"Could not read the audio file: {error}"

    duplicate = database.find_audio_by_hash(metadata["file_hash"])
    near_duplicate = None

    if not duplicate and metadata.get("fingerprint"):
        near_duplicate = database.find_audio_by_fingerprint(metadata["fingerprint"])

    metadata["uploaded_by"] = user_id
    metadata["stored_path"] = str(file_path)
    metadata["source_kind"] = source_kind

    audio_id = database.add_audio_file(metadata)

    try:
        segments, sr, segment_info = prepare_segments(file_path)
    except Exception as error:
        return [], f"Could not preprocess the audio: {error}"

    if not segments:
        return [], "The recording contained no usable audio."

    status = python_model.model_status()
    model_version = status["version"]

    detections = []

    for index, segment in enumerate(segments):
        info = segment_info[index] if index < len(segment_info) else None

        result = analyse_segment(
            segment,
            sr,
            settings,
            loaded_rules,
            gtm_scores=None,
            visual_name=f"audio{audio_id}_seg{index}",
        )

        detection_id = store_detection(audio_id, index, info, result, model_version)

        result["detection_id"] = detection_id
        result["audio_id"] = audio_id
        result["segment_index"] = index
        result["start_sec"] = info[0] if info else None
        result["end_sec"] = info[1] if info else None
        result["metadata"] = metadata
        result["duplicate_of"] = duplicate["audio_id"] if duplicate else None
        result["near_duplicate_of"] = near_duplicate["audio_id"] if near_duplicate else None

        detections.append(result)

    return detections, None


def apply_gtm_scores(detection_id, raw_gtm_scores):
    row = database.get_detection(detection_id)

    if row is None:
        return None, "That detection does not exist."

    try:
        python_scores = json.loads(row["python_scores"] or "{}")
    except ValueError:
        python_scores = {}

    settings = database.get_settings()
    loaded_rules = rules.load_rules()

    quality_result = {
        "quality": row["quality"] or "Acceptable",
        "notes": row["quality_notes"] or "",
        "noise_level": row["noise_level"] or 0.0,
    }

    repeat_count = count_recent_repeats(row["final_class"], settings)

    result = decision.build_decision(
        python_scores, raw_gtm_scores, quality_result, repeat_count, settings, loaded_rules
    )

    database.update_detection_gtm(
        detection_id,
        {
            "gtm_class": result["gtm_class"],
            "gtm_scores": json.dumps(result["gtm_scores"]) if result["gtm_available"] else None,
            "gtm_confidence": result["gtm_confidence"],
            "gtm_model_version": GTM_MODEL_VERSION if result["gtm_available"] else None,
            "agreement_status": result["agreement_status"],
            "confidence_difference": result["confidence_difference"],
            "top_two_margin": result["top_two_margin"],
            "final_class": result["final_class"],
            "confidence_level": result["confidence_level"],
            "severity": result["severity"],
            "alert_status": result["alert_status"],
            "recommended_action": result["recommended_action"],
            "manual_review_required": 1 if result["manual_review_required"] else 0,
            "manual_review_reason": result["manual_review_reason"],
            "status": result["status"],
        },
    )

    if result["alert_status"] == "Alert Generated":
        existing = [a for a in database.get_alerts() if a["detection_id"] == detection_id]

        if not existing:
            message = (
                f"{display_name(result['final_class'])} detected "
                f"({result['severity']}). {result['recommended_action']}"
            )
            database.add_alert(detection_id, result["severity"], message)

    return result, None
