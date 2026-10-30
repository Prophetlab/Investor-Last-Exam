"""Attribute views used by the original pure scoring functions."""

from bisect import bisect_right
from enum import StrEnum


class VerificationKindV11(StrEnum):
    DETERMINISTIC = "deterministic"
    JUDGE = "judge"
    FUTURE_TRUTH = "future_truth"


class FutureTruthScoringMethodV11(StrEnum):
    NUMERIC_LINEAR = "numeric_linear"
    NUMERIC_TOLERANCE = "numeric_tolerance"
    MULTICLASS_BRIER = "multiclass_brier"
    INTERVAL_COVERAGE = "interval_coverage"
    INTERVAL_SHARPNESS = "interval_sharpness"


class Record:
    def __init__(self, data):
        self.data = data

    def __getattr__(self, key):
        value = self.data.get(key)
        if key == "scoring_method":
            return FutureTruthScoringMethodV11(value)
        if key in ("parameters", "probability_output_paths", "structured_output"):
            return value
        if isinstance(value, dict):
            return Record(value)
        return value

    def model_dump(self, **kwargs):
        return self.data

    def label_for(self, value):
        return self.data["labels"][bisect_right(self.data["cutpoints"], value)]
