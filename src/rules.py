import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import ALERT_RULES_FILE

FALLBACK_RULES = {
    "rules_version": "fallback",
    "defaults": {
        "min_confidence": 0.6,
        "top_two_margin": 0.15,
        "required_consecutive_detections": 1,
        "require_model_agreement": False,
        "minimum_quality": "Poor",
        "severity": "Low",
        "recommended_action": "Record the event and continue monitoring.",
        "manual_review_on_disagreement": True,
        "escalate_after_repeats": 0,
    },
    "quality_order": ["Unusable", "Poor", "Acceptable", "Good"],
    "severity_order": ["Informational", "Low", "Medium", "High", "Critical"],
    "classes": {},
}


def load_rules():
    if not os.path.exists(ALERT_RULES_FILE):
        return FALLBACK_RULES

    try:
        with open(ALERT_RULES_FILE) as rules_file:
            rules = json.load(rules_file)
    except (ValueError, OSError):
        return FALLBACK_RULES

    if not isinstance(rules, dict) or "defaults" not in rules:
        return FALLBACK_RULES

    rules.setdefault("classes", {})
    rules.setdefault("quality_order", FALLBACK_RULES["quality_order"])
    rules.setdefault("severity_order", FALLBACK_RULES["severity_order"])

    return rules


def rule_for_class(rules, class_id):
    rule = dict(rules.get("defaults", {}))
    rule.update(rules.get("classes", {}).get(class_id, {}))
    return rule


def quality_is_enough(rules, quality, minimum_quality):
    order = rules.get("quality_order", FALLBACK_RULES["quality_order"])

    if quality not in order or minimum_quality not in order:
        return True

    return order.index(quality) >= order.index(minimum_quality)


def highest_severity(rules, first, second):
    order = rules.get("severity_order", FALLBACK_RULES["severity_order"])

    if first not in order:
        return second

    if second not in order:
        return first

    return first if order.index(first) >= order.index(second) else second


def evaluate(rules, class_id, comparison, quality_result, repeat_count, settings):
    rule = rule_for_class(rules, class_id)

    confidence = comparison["python_confidence"]
    margin = comparison["top_two_margin"]
    quality = quality_result["quality"]

    reasons = []
    conditions_met = True

    if confidence < rule["min_confidence"]:
        conditions_met = False
        reasons.append(
            f"Confidence {confidence:.2f} is below the required {rule['min_confidence']:.2f}."
        )

    if margin < rule["top_two_margin"]:
        conditions_met = False
        reasons.append(
            f"The top two classes are only {margin:.2f} apart, "
            f"{rule['top_two_margin']:.2f} is required."
        )

    if not quality_is_enough(rules, quality, rule["minimum_quality"]):
        conditions_met = False
        reasons.append(
            f"Audio quality is {quality}, at least {rule['minimum_quality']} is required."
        )

    if rule["require_model_agreement"]:
        if not comparison["gtm_available"]:
            conditions_met = False
            reasons.append("The GTM result has not been received yet.")
        elif not comparison["classes_match"]:
            conditions_met = False
            reasons.append("The two models predicted different classes.")

    if repeat_count < rule["required_consecutive_detections"]:
        conditions_met = False
        reasons.append(
            f"Detected {repeat_count} time(s), "
            f"{rule['required_consecutive_detections']} required for confirmation."
        )

    severity = rule["severity"]

    if rule.get("severity_when_repeated") and repeat_count >= rule["required_consecutive_detections"] + 1:
        severity = highest_severity(rules, severity, rule["severity_when_repeated"])

    if rule.get("noise_level_alert") is not None:
        if quality_result.get("noise_level", 0.0) > rule["noise_level_alert"]:
            severity = highest_severity(
                rules, severity, rule.get("severity_when_noisy", severity)
            )
        elif class_id == "background_noise":
            conditions_met = conditions_met and False
            reasons.append("Background noise is treated as non-critical.")

    if conditions_met:
        alert_status = "Alert Generated"
    else:
        alert_status = "No Alert"

    escalate_after = rule.get("escalate_after_repeats", 0)
    escalated = bool(escalate_after) and repeat_count > escalate_after

    return {
        "severity": severity,
        "alert_status": alert_status,
        "conditions_met": conditions_met,
        "reasons": reasons,
        "recommended_action": rule["recommended_action"],
        "manual_review_on_disagreement": rule["manual_review_on_disagreement"],
        "escalated": escalated,
        "rule_used": rule,
        "confusable_with": rule.get("confusable_with", []),
    }
