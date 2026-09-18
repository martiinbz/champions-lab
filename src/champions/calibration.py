"""Validated outcome calibration for coherent football score distributions."""
from __future__ import annotations

import copy

import numpy as np
from scipy.optimize import minimize

OUTCOMES = ("home", "draw", "away")


def _normalized_vector(probabilities) -> np.ndarray:
    values = np.asarray(list(probabilities.values()) if isinstance(probabilities, dict) else probabilities, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or (values <= 0).any() or not np.isclose(values.sum(), 1.0):
        raise ValueError("outcome probabilities must be three finite positive values summing to one")
    return values


def _normalized_matrix(probabilities) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1:] != (3,) or not np.isfinite(values).all() or (values <= 0).any() or not np.allclose(values.sum(axis=1), 1.0):
        raise ValueError("outcome probability matrix must be finite, positive and normalized")
    return values


def _outcome_indices(outcomes, size: int) -> np.ndarray:
    values = np.asarray(outcomes, dtype=int)
    if values.shape != (size,) or not np.isin(values, (0, 1, 2)).all():
        raise ValueError("outcomes must contain one home/draw/away index per probability row")
    return values


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.asarray(values, dtype=float) - np.max(values)
    result = np.exp(shifted)
    return result / result.sum()


def grid_outcomes(grid: np.ndarray) -> np.ndarray:
    """Aggregate a normalized score grid into home/draw/away probabilities."""
    values = np.asarray(grid, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1.0):
        raise ValueError("score grid must be square, finite, nonnegative and normalized")
    return np.array([np.tril(values, -1).sum(), np.trace(values), np.triu(values, 1).sum()])


class OutcomeCalibrator:
    """Temperature and class-offset calibration of a 1X2 probability vector."""

    def __init__(self, log_temperature: float = 0.0, offsets=(0.0, 0.0, 0.0)):
        self.log_temperature = float(log_temperature)
        self.offsets = np.asarray(offsets, dtype=float)
        if not np.isfinite(self.log_temperature) or self.offsets.shape != (3,) or not np.isfinite(self.offsets).all():
            raise ValueError("calibrator parameters must be finite")
        if not np.isclose(self.offsets.sum(), 0.0):
            raise ValueError("calibrator offsets must sum to zero")

    def transform(self, probabilities) -> dict[str, float]:
        base = _normalized_vector(probabilities)
        values = _softmax(np.log(base) / np.exp(self.log_temperature) + self.offsets)
        return dict(zip(OUTCOMES, map(float, values)))

    def transform_matrix(self, probabilities) -> np.ndarray:
        base = _normalized_matrix(probabilities)
        logits = np.log(base) / np.exp(self.log_temperature) + self.offsets
        logits -= logits.max(axis=1, keepdims=True)
        result = np.exp(logits)
        return result / result.sum(axis=1, keepdims=True)

    @staticmethod
    def log_loss(probabilities, outcomes) -> float:
        matrix = _normalized_matrix(probabilities)
        target = _outcome_indices(outcomes, len(matrix))
        return float(-np.log(matrix[np.arange(len(matrix)), target]).mean())

    @classmethod
    def fit(cls, probabilities, outcomes, *, l2: float = 1.0, min_samples: int = 60) -> "OutcomeCalibrator":
        matrix = _normalized_matrix(probabilities)
        target = _outcome_indices(outcomes, len(matrix))
        if not np.isfinite(l2) or l2 < 0 or isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 1:
            raise ValueError("calibration hyperparameters are invalid")
        if len(matrix) < min_samples or len(np.unique(target)) < 3:
            raise ValueError("calibration requires enough observations of home, draw and away outcomes")

        def objective(values) -> float:
            candidate = cls(values[0], [values[1], values[2], -values[1] - values[2]])
            return cls.log_loss(candidate.transform_matrix(matrix), target) + l2 * float(np.dot(values, values)) / len(matrix)

        result = minimize(objective, np.zeros(3), method="L-BFGS-B",
                          bounds=[(-1.0, 1.0), (-2.0, 2.0), (-2.0, 2.0)])
        if not result.success or not np.isfinite(result.x).all():
            raise ValueError("calibration optimization did not converge")
        return cls(result.x[0], [result.x[1], result.x[2], -result.x[1] - result.x[2]])

    def to_dict(self) -> dict:
        return {"schema_version": 1, "model_type": type(self).__name__,
                "log_temperature": self.log_temperature, "offsets": self.offsets.tolist()}

    @classmethod
    def from_dict(cls, payload: dict) -> "OutcomeCalibrator":
        data = copy.deepcopy(payload)
        if data.get("schema_version") != 1 or data.get("model_type") != cls.__name__:
            raise ValueError("unsupported calibration schema")
        return cls(data.get("log_temperature"), data.get("offsets"))


class CalibratedModel:
    """A score-distribution model whose three match outcomes are calibrated."""

    def __init__(self, base_model, calibrator: OutcomeCalibrator):
        if not hasattr(base_model, "score_probabilities") or not hasattr(base_model, "predict_goals") or not hasattr(base_model, "to_dict"):
            raise ValueError("base model must expose score probabilities, goal rates and serialization")
        if not isinstance(calibrator, OutcomeCalibrator):
            raise ValueError("calibrator must be an OutcomeCalibrator")
        self.base_model = base_model
        self.calibrator = calibrator
        self.warnings = getattr(base_model, "warnings", [])
        self.coverage = getattr(base_model, "coverage", {})
        self.params = getattr(base_model, "params", {})
        self.fit_info = {"method": "outcome_calibrated_score_distribution",
                         "base": getattr(base_model, "fit_info", {})}

    def predict_goals(self, home: str, away: str, neutral: bool = False):
        return self.base_model.predict_goals(home, away, neutral=neutral)

    def score_probabilities(self, home: str, away: str, neutral: bool = False, max_goals: int = 12) -> np.ndarray:
        base = np.asarray(self.base_model.score_probabilities(home, away, neutral=neutral, max_goals=max_goals), dtype=float)
        before = grid_outcomes(base)
        after = np.asarray(list(self.calibrator.transform(before).values()))
        multiplier = np.divide(after, before, out=np.zeros(3), where=before > 0)
        result = base.copy()
        result[np.tril_indices_from(result, -1)] *= multiplier[0]
        result[np.diag_indices_from(result)] *= multiplier[1]
        result[np.triu_indices_from(result, 1)] *= multiplier[2]
        if not np.isfinite(result).all() or result.sum() <= 0:
            raise ValueError("calibration produced an invalid score distribution")
        return result / result.sum()

    def to_dict(self) -> dict:
        return {"schema_version": 1, "model_type": type(self).__name__,
                "base_model": self.base_model.to_dict(), "calibrator": self.calibrator.to_dict(),
                "fit_info": copy.deepcopy(self.fit_info), "warnings": copy.deepcopy(self.warnings)}

    @classmethod
    def from_dict(cls, payload: dict, base_loader) -> "CalibratedModel":
        data = copy.deepcopy(payload)
        if data.get("schema_version") != 1 or data.get("model_type") != cls.__name__:
            raise ValueError("unsupported calibrated model schema")
        model = cls(base_loader(data["base_model"]), OutcomeCalibrator.from_dict(data["calibrator"]))
        model.fit_info = data.get("fit_info", model.fit_info)
        model.warnings = data.get("warnings", model.warnings)
        return model
