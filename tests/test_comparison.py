from src.comparison import (
    clean_scores,
    compare_models,
    confidence_level,
    find_overlapping,
    normalise_class_id,
    top_prediction,
    top_two_margin,
)


def test_canonical_id_passes_through():
    assert normalise_class_id("gunshot") == "gunshot"


def test_display_name_maps_to_canonical_id():
    assert normalise_class_id("Glass Breaking") == "glass_breaking"


def test_gtm_spelling_variants_map_correctly():
    assert normalise_class_id("Alarm or Siren") == "alarm_siren"
    assert normalise_class_id("Person Asking for Help") == "person_asking_help"
    assert normalise_class_id("Aggression or Violent Conflict") == "aggression"


def test_unknown_label_is_ignored():
    assert normalise_class_id("Spaceship") is None


def test_clean_scores_fills_every_class():
    scores = clean_scores({"Gunshot": 1.0})
    assert len(scores) == 10
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_clean_scores_drops_unknown_labels():
    scores = clean_scores({"Gunshot": 0.5, "Spaceship": 0.5})
    assert scores["gunshot"] == 1.0


def test_clean_scores_handles_rubbish_values():
    scores = clean_scores({"Gunshot": "not a number", "Background Noise": 1.0})
    assert scores["background_noise"] == 1.0


def test_clean_scores_handles_a_non_dict():
    assert sum(clean_scores("nonsense").values()) == 0.0


def test_top_prediction_on_empty_scores():
    name, confidence = top_prediction({name: 0.0 for name in ["gunshot"]})
    assert name is None
    assert confidence == 0.0


def test_top_two_margin_is_the_gap():
    margin = top_two_margin({"gunshot": 0.7, "aggression": 0.2, "background_noise": 0.1})
    assert abs(margin - 0.5) < 1e-6


def test_overlapping_needs_two_significant_classes():
    assert find_overlapping({"gunshot": 0.9, "aggression": 0.1}, 0.25) == []
    assert len(find_overlapping({"gunshot": 0.5, "aggression": 0.4}, 0.25)) == 2


def test_matching_models_give_an_acceptable_match(settings):
    result = compare_models(
        {"gunshot": 0.9, "background_noise": 0.1},
        {"Gunshot": 0.85, "Background Noise": 0.15},
        settings,
    )
    assert result["classes_match"]
    assert result["agreement_status"] == "Acceptable Match"


def test_different_predictions_give_a_disagreement(settings):
    result = compare_models(
        {"gunshot": 0.9, "background_noise": 0.1},
        {"Vehicle Horn": 0.8, "Gunshot": 0.2},
        settings,
    )
    assert not result["classes_match"]
    assert result["agreement_status"] == "Model Disagreement"


def test_low_confidence_gives_an_uncertain_result(settings):
    result = compare_models(
        {"gunshot": 0.45, "glass_breaking": 0.40, "background_noise": 0.15},
        {"Gunshot": 0.42, "Glass Breaking": 0.38, "Background Noise": 0.20},
        settings,
    )
    assert result["agreement_status"] == "Uncertain Result"


def test_missing_gtm_is_reported_not_guessed(settings):
    result = compare_models({"gunshot": 0.9, "background_noise": 0.1}, {}, settings)
    assert not result["gtm_available"]
    assert result["agreement_status"] == "Awaiting GTM"
    assert result["confidence_difference"] is None


def test_confidence_difference_is_the_absolute_gap(settings):
    result = compare_models(
        {"gunshot": 0.9, "background_noise": 0.1},
        {"Gunshot": 0.7, "Background Noise": 0.3},
        settings,
    )
    assert abs(result["confidence_difference"] - 0.2) < 1e-6


def test_top_three_is_returned_for_both_models(settings):
    result = compare_models(
        {"gunshot": 0.5, "aggression": 0.3, "panic_scream": 0.15, "background_noise": 0.05},
        {"Gunshot": 0.5, "Aggression": 0.3, "Panic Scream": 0.2},
        settings,
    )
    assert len(result["python_top3"]) == 3
    assert len(result["gtm_top3"]) == 3


def test_confidence_level_bands(settings):
    assert confidence_level(0.95, settings) == "High"
    assert confidence_level(0.7, settings) == "Medium"
    assert confidence_level(0.3, settings) == "Low"
