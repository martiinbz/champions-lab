import json

import numpy as np
import pytest

from champions.ensemble import EnsembleModel, model_from_dict


class FixedModel:
    def __init__(self, grid, rates=(1.0, 1.0)):
        self.grid = np.asarray(grid, dtype=float)
        self.rates = rates

    def score_probabilities(self, home, away, neutral=False, max_goals=12):
        return self.grid

    def predict_goals(self, home, away, neutral=False):
        return self.rates

    def to_dict(self):
        return {"model_type": "FixedModel"}


def test_ensemble_mixes_score_probabilities_and_expected_goal_rates():
    home = np.zeros((2, 2)); home[1, 0] = 1.0
    away = np.zeros((2, 2)); away[0, 1] = 1.0
    model = EnsembleModel([FixedModel(home, (2.0, 0.5)), FixedModel(away, (0.5, 2.0))], [0.25, 0.75])

    np.testing.assert_allclose(model.score_probabilities("A", "B"), 0.25 * home + 0.75 * away)
    assert model.predict_goals("A", "B") == pytest.approx((0.875, 1.625))


def test_ensemble_sampling_is_seeded_and_serialization_rejects_invalid_weights():
    grid = np.eye(2) / 2
    model = EnsembleModel([FixedModel(grid), FixedModel(grid)], [0.5, 0.5])
    assert np.array_equal(
        model.sample_scores("A", "B", 10, np.random.default_rng(2)),
        model.sample_scores("A", "B", 10, np.random.default_rng(2)),
    )
    with pytest.raises(ValueError):
        EnsembleModel([FixedModel(grid)], [0.9])


def test_model_from_dict_restores_goal_model_payload():
    from champions.models import GoalModel
    import pandas as pd
    history = pd.DataFrame([{
        "date": "2025-01-01T00:00:00Z", "home": "A", "away": "B", "home_goals": 1, "away_goals": 0,
    }])
    payload = GoalModel().fit(history, "2026-01-01").to_dict()
    restored = model_from_dict(json.loads(json.dumps(payload)))
    assert type(restored).__name__ == "GoalModel"
