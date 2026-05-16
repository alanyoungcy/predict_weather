"""Probability calibration helpers."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class ProbabilityCalibrator:
    def __init__(self, isotonic: IsotonicRegression):
        self.isotonic = isotonic

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        probs = np.asarray(probabilities, dtype='float64')
        return np.clip(self.isotonic.predict(probs), 0.0, 1.0)



def fit_isotonic_calibrator(probabilities: np.ndarray, outcomes: np.ndarray) -> ProbabilityCalibrator:
    probs = np.asarray(probabilities, dtype='float64')
    obs = np.asarray(outcomes, dtype='float64')
    isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    isotonic.fit(probs, obs)
    return ProbabilityCalibrator(isotonic)
