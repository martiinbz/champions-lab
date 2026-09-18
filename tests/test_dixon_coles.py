import json

import numpy as np
import pandas as pd
import pytest

from champions.dixon_coles import DixonColesModel, dixon_coles_grid
from champions.models import independent_poisson_grid


def _history(n=100):
    rows = []
    teams = ["A", "B", "C", "D"]
    for i in range(n):
        home, away = teams[i % 4], teams[(i + 1) % 4]
        rows.append({
            "date": (pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=i)).isoformat(),
            "home": home, "away": away, "home_goals": (i + (home == "A")) % 4,
            "away_goals": i % 3, "competition": "E0" if i % 5 else "UCL", "source": "test",
        })
    return pd.DataFrame(rows)


def test_dixon_coles_changes_low_score_cells_but_keeps_grid_normalized():
    base = independent_poisson_grid(1.4, 1.1, 12)
    adjusted = dixon_coles_grid(1.4, 1.1, rho=-0.08, max_goals=12)

    assert adjusted.sum() == pytest.approx(1.0)
    assert adjusted[0, 0] != pytest.approx(base[0, 0])
    assert adjusted[1, 1] != pytest.approx(base[1, 1])
    assert adjusted[3, 2] / adjusted[2, 3] == pytest.approx(base[3, 2] / base[2, 3])


def test_dixon_coles_roundtrip_preserves_probabilities_and_records_ratings():
    model = DixonColesModel(max_iter=100).fit(_history(), "2026-01-01")
    restored = DixonColesModel.from_dict(json.loads(json.dumps(model.to_dict())))

    np.testing.assert_allclose(
        model.score_probabilities("A", "B"), restored.score_probabilities("A", "B")
    )
    assert "rating_config" in model.fit_info
    assert np.isfinite(model.rho)


def test_dixon_coles_unknown_team_has_finite_prior_sensitive_prediction():
    model = DixonColesModel(max_iter=100).fit(_history(), "2026-01-01")

    grid = model.score_probabilities("new home", "new away")

    assert grid.shape == (13, 13)
    assert np.isfinite(grid).all()
    assert grid.sum() == pytest.approx(1.0)
