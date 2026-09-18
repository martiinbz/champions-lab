"""Serializable mixtures of coherent football score distributions."""
from __future__ import annotations

import copy

import numpy as np


class EnsembleModel:
    """Weighted mixture of independently fitted score-distribution models."""

    def __init__(self, components: list, weights):
        values = np.asarray(weights, dtype=float)
        if not components or len(components) != len(values) or not np.isfinite(values).all():
            raise ValueError("components and finite weights must have the same nonzero length")
        if (values < 0).any() or not np.isclose(values.sum(), 1.0):
            raise ValueError("ensemble weights must be nonnegative and sum to one")
        if any(not hasattr(component, "score_probabilities") for component in components):
            raise ValueError("each ensemble component must expose score_probabilities")
        self.components = list(components)
        self.weights = values
        self.warnings = []
        self.fit_info = {"method": "weighted_score_distribution_ensemble", "success": True}
        reference = self.components[0]
        self.coverage = getattr(reference, "coverage", {})
        self.params = getattr(reference, "params", {"min_team_matches": 1})

    def score_probabilities(self, home: str, away: str, neutral: bool = False,
                            max_goals: int = 12) -> np.ndarray:
        grids = [np.asarray(component.score_probabilities(home, away, neutral=neutral, max_goals=max_goals), dtype=float)
                 for component in self.components]
        if any(grid.shape != grids[0].shape or not np.isfinite(grid).all() or (grid < 0).any()
               or not np.isclose(grid.sum(), 1.0) for grid in grids):
            raise ValueError("component returned an invalid score probability grid")
        mixed = sum(weight * grid for weight, grid in zip(self.weights, grids))
        return mixed / mixed.sum()

    def predict_goals(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        rates = np.asarray([component.predict_goals(home, away, neutral=neutral) for component in self.components], dtype=float)
        if rates.shape != (len(self.components), 2) or not np.isfinite(rates).all() or (rates <= 0).any():
            raise ValueError("component returned invalid goal rates")
        return tuple(map(float, self.weights @ rates))

    def sample_scores(self, home: str, away: str, size: int, rng: np.random.Generator,
                      neutral: bool = False) -> np.ndarray:
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1:
            raise ValueError("size must be a positive integer")
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy Generator")
        grid = self.score_probabilities(home, away, neutral=neutral)
        selected = rng.choice(grid.size, size=int(size), p=grid.ravel())
        return np.column_stack(np.unravel_index(selected, grid.shape)).astype(np.int64, copy=False)

    def to_dict(self) -> dict:
        if any(not hasattr(component, "to_dict") for component in self.components):
            raise RuntimeError("all ensemble components must be serializable")
        return copy.deepcopy({
            "schema_version": 1,
            "model_type": type(self).__name__,
            "weights": self.weights.tolist(),
            "components": [component.to_dict() for component in self.components],
            "fit_info": self.fit_info,
            "warnings": self.warnings,
        })

    @classmethod
    def from_dict(cls, payload: dict) -> "EnsembleModel":
        data = copy.deepcopy(payload)
        if data.get("schema_version") != 1 or data.get("model_type") != cls.__name__:
            raise ValueError("unsupported schema_version or incorrect model_type")
        model = cls([model_from_dict(component) for component in data.get("components", [])], data.get("weights", []))
        model.fit_info = data.get("fit_info", model.fit_info)
        model.warnings = data.get("warnings", [])
        return model


def model_from_dict(payload: dict):
    """Restore any production score model stored in an immutable snapshot."""
    model_type = payload.get("model_type")
    if model_type == "GoalModel":
        from champions.models import GoalModel
        return GoalModel.from_dict(payload)
    if model_type == "BaselineModel":
        from champions.models import BaselineModel
        return BaselineModel.from_dict(payload)
    if model_type == "DixonColesModel":
        from champions.dixon_coles import DixonColesModel
        return DixonColesModel.from_dict(payload)
    if model_type == "EnsembleModel":
        return EnsembleModel.from_dict(payload)
    raise ValueError("unsupported serialized model type")
