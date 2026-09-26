from src.comparison import compare_models
from src.rules import evaluate, load_rules, quality_is_enough, rule_for_class


def test_rules_file_loads():
    rules = load_rules()
    assert rules["classes"]
    assert "gunshot" in rules["classes"]


def test_every_mandatory_class_has_a_rule():
    from config.config import CLASSES

    rules = load_rules()
    for class_id in CLASSES:
        assert class_id in rules["classes"], class_id


def test_defaults_fill_missing_keys():
    rules = load_rules()
    rule = rule_for_class(rules, "vehicle_horn")
    assert "min_confidence" in rule
    assert "severity" in rule


def test_unknown_class_falls_back_to_defaults():
    rules = load_rules()
    rule = rule_for_class(rules, "not_a_class")
    assert rule["severity"] == rules["defaults"]["severity"]


def test_quality_ordering():
    rules = load_rules()
    assert quality_is_enough(rules, "Good", "Acceptable")
    assert not quality_is_enough(rules, "Poor", "Acceptable")


def test_gunshot_needs_agreement_and_repeats(settings, good_quality):
    rules = load_rules()
    comparison = compare_models(
        {"gunshot": 0.95, "background_noise": 0.05},
        {"Gunshot": 0.92, "Background Noise": 0.08},
        settings,
    )

    single = evaluate(rules, "gunshot", comparison, good_quality, 1, settings)
    assert not single["conditions_met"]

    confirmed = evaluate(rules, "gunshot", comparison, good_quality, 2, settings)
    assert confirmed["conditions_met"]
    assert confirmed["alert_status"] == "Alert Generated"
    assert confirmed["severity"] == "Critical"


def test_gunshot_without_agreement_does_not_alert(settings, good_quality):
    rules = load_rules()
    comparison = compare_models(
        {"gunshot": 0.95, "background_noise": 0.05},
        {"Vehicle Horn": 0.8, "Gunshot": 0.2},
        settings,
    )
    result = evaluate(rules, "gunshot", comparison, good_quality, 3, settings)
    assert not result["conditions_met"]
    assert result["alert_status"] == "No Alert"


def test_poor_quality_blocks_a_critical_alert(settings):
    rules = load_rules()
    comparison = compare_models(
        {"panic_scream": 0.95, "background_noise": 0.05},
        {"Panic Scream": 0.93, "Background Noise": 0.07},
        settings,
    )
    poor = {"quality": "Poor", "notes": "noisy", "noise_level": 0.7}
    result = evaluate(rules, "panic_scream", comparison, poor, 2, settings)
    assert not result["conditions_met"]


def test_low_confidence_blocks_an_alert(settings, good_quality):
    rules = load_rules()
    comparison = compare_models(
        {"glass_breaking": 0.4, "background_noise": 0.6},
        {"Background Noise": 0.7, "Glass Breaking": 0.3},
        settings,
    )
    result = evaluate(rules, "glass_breaking", comparison, good_quality, 2, settings)
    assert not result["conditions_met"]


def test_background_noise_is_not_critical(settings, good_quality):
    rules = load_rules()
    comparison = compare_models(
        {"background_noise": 0.95, "vehicle_horn": 0.05},
        {"Background Noise": 0.93, "Vehicle Horn": 0.07},
        settings,
    )
    result = evaluate(rules, "background_noise", comparison, good_quality, 1, settings)
    assert result["severity"] == "Informational"


def test_noisy_background_raises_severity(settings):
    rules = load_rules()
    comparison = compare_models(
        {"background_noise": 0.95, "vehicle_horn": 0.05},
        {"Background Noise": 0.93, "Vehicle Horn": 0.07},
        settings,
    )
    noisy = {"quality": "Acceptable", "notes": "", "noise_level": 0.8}
    result = evaluate(rules, "background_noise", comparison, noisy, 1, settings)
    assert result["severity"] == "Medium"


def test_aggression_escalates_when_repeated(settings, good_quality):
    rules = load_rules()
    comparison = compare_models(
        {"aggression": 0.9, "background_noise": 0.1},
        {"Aggression": 0.88, "Background Noise": 0.12},
        settings,
    )
    repeated = evaluate(rules, "aggression", comparison, good_quality, 4, settings)
    assert repeated["severity"] == "Critical"


def test_every_rule_reports_a_reason_when_blocked(settings, good_quality):
    rules = load_rules()
    comparison = compare_models({"gunshot": 0.3, "background_noise": 0.7}, {}, settings)
    result = evaluate(rules, "gunshot", comparison, good_quality, 1, settings)
    assert result["reasons"]
