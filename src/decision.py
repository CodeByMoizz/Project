import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import display_name
from src import comparison as comparison_module
from src import rules as rules_module


# Collects every manual-review condition the SRS lists.
def review_reasons(comparison, quality_result, rule_result, settings):
    reasons = []

    if comparison["gtm_available"] and not comparison["classes_match"]:
        reasons.append("The Python and GTM models predicted different classes.")

    if comparison["python_confidence"] < settings.get("min_confidence", 0.6):
        reasons.append("The prediction confidence is below the configured minimum.")

    if quality_result["quality"] in ("Poor", "Unusable"):
        reasons.append(f"Audio quality is {quality_result['quality']}.")

    if comparison["top_two_margin"] < settings.get("top_two_margin", 0.15):
        reasons.append("The top two classes have similar confidence scores.")

    if len(comparison["overlapping_classes"]) > 1:
        names = ", ".join(display_name(c) for c in comparison["overlapping_classes"])
        reasons.append(f"Possible overlapping sounds: {names}.")

    if comparison["python_class"] is None:
        reasons.append("The sound does not match any supported pattern.")

    critical_severities = ("High", "Critical")
    if rule_result["severity"] in critical_severities and not rule_result["conditions_met"]:
        if comparison["gtm_available"] and not comparison["classes_match"]:
            reasons.append("A critical event was detected without model agreement.")
        else:
            reasons.append("A critical event did not meet its confirmation rules, possible false alarm.")

    return reasons


def build_decision(python_scores, gtm_scores, quality_result, repeat_count, settings, rules=None):
    if rules is None:
        rules = rules_module.load_rules()

    comparison = comparison_module.compare_models(python_scores, gtm_scores, settings)

    detected_class = comparison["python_class"] or "unknown"

    rule_result = rules_module.evaluate(
        rules, detected_class, comparison, quality_result, repeat_count, settings
    )

    reasons = review_reasons(comparison, quality_result, rule_result, settings)

    if reasons and not rule_result["conditions_met"]:
        final_class = "unknown"
    else:
        final_class = detected_class

    manual_review_required = bool(reasons)

    if manual_review_required:
        status = "Manual Review"
    elif rule_result["alert_status"] == "Alert Generated":
        status = "Alert Generated"
    elif final_class == "unknown":
        status = "Uncertain"
    else:
        status = "Classified"

    decision = dict(comparison)
    decision.update(
        {
            "detected_class": detected_class,
            "detected_class_name": display_name(detected_class),
            "final_class": final_class,
            "final_class_name": display_name(final_class),
            "confidence_level": comparison_module.confidence_level(
                comparison["python_confidence"], settings
            ),
            "quality": quality_result["quality"],
            "quality_notes": quality_result["notes"],
            "noise_level": quality_result["noise_level"],
            "severity": rule_result["severity"],
            "alert_status": rule_result["alert_status"],
            "recommended_action": rule_result["recommended_action"],
            "rule_reasons": rule_result["reasons"],
            "escalated": rule_result["escalated"],
            "confusable_with": rule_result["confusable_with"],
            "manual_review_required": manual_review_required,
            "manual_review_reason": " ".join(reasons),
            "repeat_count": repeat_count,
            "status": status,
        }
    )

    return decision
