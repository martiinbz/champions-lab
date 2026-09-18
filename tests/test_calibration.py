import json

import numpy as np
import pytest

from champions.calibration import CalibratedModel, OutcomeCalibrator
from champions.ensemble import model_from_dict
from champions.models import GoalModel


class GridModel:
    def __init__(self, grid):
        self.grid = np.asarray(grid, dtype=float)
        self.warnings = []
        self.fit_info = {"method": "test"}
        self.coverage = {}
        self.params = {}

    def score_probabilities(self, home, away, neutral=False, max_goals=12):
        return self.grid

    def predict_goals(self, home, away, neutral=False):
        return (1.2, 1.0)

    def to_dict(self):
        return {"model_type": "FixedModel"}


def test_calibrator_returns_finite_normalized_three_way_probabilities():
    result = OutcomeCalibrator(np.log(1.2), [.1, -.2, .1]).transform([.62, .21, .17])
    assert set(result) == {"home", "draw", "away"}
    assert sum(result.values()) == pytest.approx(1.0)
    assert all(0 < value < 1 for value in result.values())


def test_calibrated_model_reweights_grid_by_result_without_breaking_score_support():
    grid = np.array([[.16, .08], [.18, .58]])
    model = CalibratedModel(GridModel(grid), OutcomeCalibrator(0.0, [.4, -.2, -.2]))
    calibrated = model.score_probabilities("A", "B", max_goals=1)
    assert calibrated.shape == grid.shape
    assert calibrated.sum() == pytest.approx(1.0)
    assert (calibrated >= 0).all()
    assert calibrated[1, 0] > grid[1, 0]


def test_fit_reduces_log_loss_on_a_deliberately_overconfident_training_sample():
    probabilities = np.tile([.94, .03, .03], (90, 1))
    outcomes = np.array([0] * 54 + [1] * 18 + [2] * 18)
    calibrator = OutcomeCalibrator.fit(probabilities, outcomes, l2=1.0, min_samples=60)
    raw_loss = OutcomeCalibrator.log_loss(probabilities, outcomes)
    fitted = np.array([list(calibrator.transform(row).values()) for row in probabilities])
    assert OutcomeCalibrator.log_loss(fitted, outcomes) < raw_loss


def test_calibrated_goal_model_json_round_trip_keeps_score_distribution():
    history = __import__("pandas").DataFrame([
        {"date": "2025-01-01T00:00:00Z", "home": "A", "away": "B", "home_goals": 1, "away_goals": 0},
        {"date": "2025-01-02T00:00:00Z", "home": "B", "away": "A", "home_goals": 0, "away_goals": 2},
    ])
    model = CalibratedModel(GoalModel().fit(history, "2026-01-01"), OutcomeCalibrator(.1, [.05, -.1, .05]))
    restored = model_from_dict(json.loads(json.dumps(model.to_dict())))
    np.testing.assert_allclose(restored.score_probabilities("A", "B"), model.score_probabilities("A", "B"))
