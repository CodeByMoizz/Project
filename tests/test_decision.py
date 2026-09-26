from src.decision import build_decision


def test_clear_critical_event_generates_an_alert(settings, good_quality):
    result = build_decision(
        {"gunshot": 0.95, "background_noise": 0.05},
        {"Gunshot": 0.92, "Background Noise": 0.08},
        good_quality,
        2,
        settings,
    )
    assert result["final_class"] == "gunshot"
    assert result["severity"] == "Critical"
    assert result["alert_status"] == "Alert Generated"
    assert not result["manual_review_required"]
    assert result["status"] == "Alert Generated"


def test_disagreement_goes_to_manual_review(settings, good_quality):
    result = build_decision(
        {"gunshot": 0.9, "background_noise": 0.1},
        {"Vehicle Horn": 0.85, "Gunshot": 0.15},
        good_quality,
        2,
        settings,
    )
    assert result["manual_review_required"]
    assert result["final_class"] == "unknown"
    assert result["status"] == "Manual Review"
    assert "different classes" in result["manual_review_reason"]


def test_low_confidence_goes_to_manual_review(settings, good_quality):
    result = build_decision(
        {"gunshot": 0.35, "glass_breaking": 0.33, "background_noise": 0.32},
        {"Gunshot": 0.34, "Glass Breaking": 0.33, "Background Noise": 0.33},
        good_quality,
        1,
        settings,
    )
    assert result["manual_review_required"]
    assert result["confidence_level"] == "Low"


def test_poor_quality_goes_to_manual_review(settings):
    poor = {"quality": "Poor", "notes": "clipped", "noise_level": 0.7}
    result = build_decision(
        {"glass_breaking": 0.9, "background_noise": 0.1},
        {"Glass Breaking": 0.88, "Background Noise": 0.12},
        poor,
        2,
        settings,
    )
    assert result["manual_review_required"]
    assert "quality" in result["manual_review_reason"].lower()


def test_overlapping_sounds_are_reported(settings, good_quality):
    result = build_decision(
        {"gunshot": 0.45, "glass_breaking": 0.40, "background_noise": 0.15},
        {},
        good_quality,
        1,
        settings,
    )
    assert len(result["overlapping_classes"]) > 1
    assert "overlapping" in result["manual_review_reason"].lower()


def test_model_outputs_are_preserved_even_when_final_is_unknown(settings, good_quality):
    result = build_decision(
        {"gunshot": 0.9, "background_noise": 0.1},
        {"Vehicle Horn": 0.85, "Gunshot": 0.15},
        good_quality,
        1,
        settings,
    )
    assert result["final_class"] == "unknown"
    assert result["python_class"] == "gunshot"
    assert result["gtm_class"] == "vehicle_horn"
    assert result["python_scores"]["gunshot"] == 0.9


def test_background_noise_is_classified_without_an_alert(settings, good_quality):
    result = build_decision(
        {"background_noise": 0.95, "vehicle_horn": 0.05},
        {"Background Noise": 0.94, "Vehicle Horn": 0.06},
        good_quality,
        1,
        settings,
    )
    assert result["severity"] == "Informational"
    assert result["alert_status"] == "No Alert"


def test_a_decision_always_has_the_required_fields(settings, good_quality):
    result = build_decision({"vehicle_horn": 0.8, "background_noise": 0.2}, {}, good_quality, 1, settings)

    for field in [
        "final_class",
        "agreement_status",
        "confidence_level",
        "severity",
        "alert_status",
        "recommended_action",
        "manual_review_required",
        "top_two_margin",
    ]:
        assert field in result, field


def test_empty_scores_do_not_raise(settings, good_quality):
    result = build_decision({}, {}, good_quality, 1, settings)
    assert result["final_class"] == "unknown"
    assert result["manual_review_required"]
