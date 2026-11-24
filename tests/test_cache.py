from faultwatch.cache import prediction_key


def test_prediction_key_changes_with_readings_order_and_model():
    first = {"engine_id": 1, "cycle": 8, "settings": [0.0] * 3, "sensors": [0.0] * 21}
    second = {**first, "cycle": 9}
    key = prediction_key("version", "artifact", [first, second])
    assert prediction_key("version", "artifact", [dict(first), dict(second)]) == key
    assert prediction_key("version", "artifact", [second, first]) != key
    assert prediction_key("version", "another-model", [first, second]) != key
    changed_reading = {**second, "sensors": [1.0] * 21}
    assert prediction_key("version", "artifact", [first, changed_reading]) != key
