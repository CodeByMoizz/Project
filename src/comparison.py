import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import CLASSES

# This module only reads the two finished score sets. Neither model's output is
# ever passed into the other one, they are compared after the fact.


def normalise_class_id(name):
    if not name:
        return None

    cleaned = str(name).strip().lower().replace("-", " ").replace("_", " ")
    cleaned = " ".join(cleaned.split())

    direct = cleaned.replace(" ", "_")
    if direct in CLASSES:
        return direct

    # Teachable Machine class names are free text, so a few spellings are mapped
    # back onto the canonical ids.
    aliases = {
        "machinery fault": "machinery_fault",
        "glass breaking": "glass_breaking",
        "glass break": "glass_breaking",
        "alarm or siren": "alarm_siren",
        "alarm siren": "alarm_siren",
        "alarm": "alarm_siren",
        "siren": "alarm_siren",
        "vehicle horn": "vehicle_horn",
        "horn": "vehicle_horn",
        "animal sound": "animal_sound",
        "animal": "animal_sound",
        "gunshot": "gunshot",
        "gun shot": "gunshot",
        "panic scream": "panic_scream",
        "scream": "panic_scream",
        "aggression": "aggression",
        "aggression or violent conflict": "aggression",
        "violent conflict": "aggression",
        "person asking for help": "person_asking_help",
        "person asking help": "person_asking_help",
        "asking for help": "person_asking_help",
        "help": "person_asking_help",
        "background noise": "background_noise",
        "background": "background_noise",
        "noise": "background_noise",
    }

    return aliases.get(cleaned)


def clean_scores(raw_scores):
    scores = {name: 0.0 for name in CLASSES}

    if not isinstance(raw_scores, dict):
        return scores

    for key, value in raw_scores.items():
        class_id = normalise_class_id(key)

        if class_id is None:
            continue

        try:
            score = float(value)
        except (TypeError, ValueError):
            continue

        if score != score or score < 0:
            score = 0.0

        scores[class_id] = score

    total = sum(scores.values())

    # Teachable Machine already returns probabilities, but a partial or padded
    # score set has to be renormalised before it can be compared.
    if total > 0:
        scores = {name: value / total for name, value in scores.items()}

    return scores


def sorted_scores(scores):
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def top_prediction(scores):
    ranked = sorted_scores(scores)

    if not ranked or ranked[0][1] <= 0:
        return None, 0.0

    return ranked[0][0], round(ranked[0][1], 4)


def top_n(scores, count=3):
    return [(name, round(value, 4)) for name, value in sorted_scores(scores)[:count]]


def top_two_margin(scores):
    ranked = sorted_scores(scores)

    if len(ranked) < 2:
        return 0.0

    return round(ranked[0][1] - ranked[1][1], 4)


def find_overlapping(scores, threshold):
    significant = [name for name, value in sorted_scores(scores) if value >= threshold]

    if len(significant) < 2:
        return []

    return significant


def compare_models(raw_python_scores, raw_gtm_scores, settings):
    # Both score sets are mapped onto the canonical class ids first, otherwise
    # "Gunshot" from Teachable Machine would never match "gunshot".
    python_scores = clean_scores(raw_python_scores)
    gtm_scores = clean_scores(raw_gtm_scores)

    python_class, python_confidence = top_prediction(python_scores)
    gtm_class, gtm_confidence = top_prediction(gtm_scores)

    python_margin = top_two_margin(python_scores)
    gtm_margin = top_two_margin(gtm_scores)

    overlap_threshold = settings.get("overlap_confidence", 0.25)
    overlapping = find_overlapping(python_scores, overlap_threshold)

    result = {
        "python_scores": python_scores,
        "gtm_scores": gtm_scores,
        "python_class": python_class,
        "python_confidence": python_confidence,
        "python_top3": top_n(python_scores),
        "python_margin": python_margin,
        "gtm_class": gtm_class,
        "gtm_confidence": gtm_confidence,
        "gtm_top3": top_n(gtm_scores) if gtm_class else [],
        "gtm_margin": gtm_margin,
        "classes_match": bool(gtm_class) and python_class == gtm_class,
        "gtm_available": bool(gtm_class),
        "overlapping_classes": overlapping,
        # The margin used downstream is the Python model's, because the GTM
        # result may not have arrived yet.
        "top_two_margin": python_margin,
    }

    if gtm_class:
        result["confidence_difference"] = round(
            abs(python_confidence - gtm_confidence), 4
        )
    else:
        result["confidence_difference"] = None

    result["agreement_status"] = agreement_status(result, settings)

    return result


# One of the four statuses the SRS asks for.
def agreement_status(comparison, settings):
    if not comparison["gtm_available"]:
        return "Awaiting GTM"

    min_confidence = settings.get("min_confidence", 0.6)
    max_difference = settings.get("max_confidence_difference", 0.3)
    margin = settings.get("top_two_margin", 0.15)

    python_confidence = comparison["python_confidence"]
    gtm_confidence = comparison["gtm_confidence"]
    difference = comparison["confidence_difference"]

    if not comparison["classes_match"]:
        return "Model Disagreement"

    both_confident = (
        python_confidence >= min_confidence and gtm_confidence >= min_confidence
    )
    clear_margin = comparison["python_margin"] >= margin

    if not both_confident or comparison["python_margin"] < margin * 0.5:
        return "Uncertain Result"

    if difference <= max_difference and clear_margin:
        return "Acceptable Match"

    return "Weak Match"


def confidence_level(confidence, settings):
    min_confidence = settings.get("min_confidence", 0.6)

    if confidence >= 0.85:
        return "High"

    if confidence >= min_confidence:
        return "Medium"

    return "Low"
