"""Behavioral tests for the public goal-model API."""
import importlib.util
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest


def module():
    path = Path(__file__).parents[1] / "src/champions/models.py"
    assert path.exists(), "goal model implementation is missing"
    spec = importlib.util.spec_from_file_location("champions_models", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def history(n=360):
    rng = np.random.default_rng(17)
    teams = ["strong", "average", "weak", "other"]
    attack = [0.65, 0, -0.65, 0]
    rows = []
    for i in range(n):
        h, a = rng.choice(4, 2, replace=False)
        rows.append((pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                     teams[h], teams[a], rng.poisson(np.exp(.2 + .25 + attack[h])),
                     rng.poisson(np.exp(.2 + attack[a]))))
    return pd.DataFrame(rows, columns=["date", "home", "away", "home_goals", "away_goals"])


def test_learns_strength_and_home_advantage():
    m = module().GoalModel().fit(history(), "2022-01-01")
    assert m.predict_goals("strong", "average")[0] > 1.5 * m.predict_goals("weak", "average")[0]
    assert m.predict_goals("strong", "average")[0] > m.predict_goals("strong", "average", neutral=True)[0]


@pytest.mark.parametrize("kind", ["GoalModel", "BaselineModel"])
def test_strict_cutoff_and_incomplete_exclusion(kind):
    cls = getattr(module(), kind)
    data = history(80)
    cutoff = "2020-03-01"
    clean = cls().fit(data.iloc[:60], cutoff)
    extra = pd.DataFrame([
        [cutoff, "future", "strong", 99, 99],
        ["2021-01-01", "future", "strong", 99, 99],
        ["2020-02-01", "pending", "strong", np.nan, 0],
    ], columns=data.columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dirty = cls().fit(pd.concat([data, extra]), cutoff)
    assert clean.to_dict() == dirty.to_dict()
    assert "future" not in dirty.to_dict()["coverage"]


def test_completion_timestamp_is_respected():
    mod = module()
    data = history(40)
    data["completed_at"] = data.date + pd.Timedelta(hours=2)
    data.loc[0, "completed_at"] = "2025-01-01"
    a = mod.GoalModel().fit(data, "2021-01-01")
    b = mod.GoalModel().fit(data.iloc[1:], "2021-01-01")
    assert a.to_dict() == b.to_dict()


@pytest.mark.parametrize("kind", ["GoalModel", "BaselineModel"])
def test_reproducible_and_json_roundtrip(kind):
    cls = getattr(module(), kind)
    data = history(100)
    a = cls().fit(data, "2021-01-01")
    b = cls().fit(data.sample(frac=1, random_state=8), "2021-01-01")
    assert a.to_dict() == b.to_dict()
    restored = cls.from_dict(json.loads(json.dumps(a.to_dict(), allow_nan=False)))
    assert restored.predict_goals("strong", "weak") == a.predict_goals("strong", "weak")
    assert a.to_dict()["n_matches"] == 100
    assert a.to_dict()["train_cutoff"].startswith("2021-01-01")


def test_empty_and_sparse_priors_warn_and_are_finite():
    mod = module()
    with pytest.warns(mod.CoverageWarning):
        empty = mod.GoalModel().fit(history(0), "2021-01-01")
    with pytest.warns(mod.CoverageWarning):
        rates = empty.predict_goals("unseen", "another")
    assert np.all(np.isfinite(rates)) and min(rates) > 0
    with pytest.warns(mod.CoverageWarning):
        m = mod.GoalModel().fit(history(1), "2021-01-01")
    with pytest.warns(mod.CoverageWarning):
        assert max(m.predict_goals("strong", "weak")) < 10


@pytest.mark.parametrize("rates", [(1.4, 1.1), (0.001, 30), (30, .001), (100, 100)])
def test_outcomes_normalized(rates):
    p = module().outcome_probabilities(*rates)
    assert set(p) == {"home", "draw", "away"}
    assert sum(p.values()) == pytest.approx(1, abs=1e-12)
    assert all(np.isfinite(x) and 0 <= x <= 1 for x in p.values())


def test_temporal_evaluation_is_insulated_from_future():
    mod = module()
    data = history()
    result = mod.evaluate(data, "2020-11-01")
    future = data.copy()
    future.loc[future.date >= "2020-11-01", "home_goals"] = 90
    assert result == mod.evaluate(future, "2020-11-01")
    assert result["status"] == "ok"
    assert result["train_last_date"] < result["holdout_start"]
    assert result["holdout_end"] < result["cutoff"]
    for metrics in result["models"].values():
        assert metrics["log_loss"] > 0
        assert 0 <= metrics["brier"] <= 2
        for bins in metrics["calibration_bins"].values():
            assert sum(b["count"] for b in bins) == result["n_holdout"]
    json.dumps(result, allow_nan=False)


def test_evaluation_never_fits_holdout(monkeypatch):
    mod = module()
    original = mod.GoalModel.fit
    seen = []
    def spy(self, data, cutoff):
        seen.append((pd.to_datetime(data.date, utc=True).max(), pd.Timestamp(cutoff)))
        return original(self, data, cutoff)
    monkeypatch.setattr(mod.GoalModel, "fit", spy)
    report = mod.evaluate(history(), "2022-01-01", max_holdout=25)
    assert report["n_holdout"] == 25
    assert seen and all(last < cutoff for last, cutoff in seen)


def test_insufficient_evaluation_and_bad_inputs():
    mod = module()
    assert mod.evaluate(history(2), "2021-01-01")["status"] == "insufficient_data"
    with pytest.raises(ValueError):
        mod.GoalModel(half_life_days=0)
    with pytest.raises(ValueError):
        mod.GoalModel().fit(history(), "not-a-date")
    with pytest.raises(ValueError):
        mod.GoalModel().fit(pd.DataFrame(), "2021-01-01")
    with pytest.raises(RuntimeError):
        mod.GoalModel().predict_goals("a", "b")


def test_pipeline_baseline_and_persistent_coverage_warnings():
    mod = module()
    model = mod.GoalModel(kind="baseline").fit(history(80), "2021-01-01")
    baseline = mod.BaselineModel().fit(history(80), "2021-01-01")
    assert model.predict_goals("strong", "weak") == baseline.predict_goals("strong", "weak")
    with pytest.warns(mod.CoverageWarning):
        model.predict_goals("new club", "strong")
    assert any("new club" in message for message in model.warnings)
    payload = json.loads(json.dumps(model.to_dict()))
    assert payload["warnings"] == model.warnings
    restored = mod.GoalModel.from_dict(payload)
    assert restored.kind == "baseline"
    assert restored.warnings == model.warnings
    report = mod.evaluate(history(100), "2021-01-01")
    assert report["selected_model"] == report["recommended_model"]
    with pytest.raises(ValueError):
        mod.GoalModel(kind="invalid")


def test_evaluation_strength_and_fallback():
    mod = module()
    report = mod.evaluate(history(800), "2025-01-01")
    assert report["selected_model"] == "poisson"
    assert report["models"]["poisson"]["log_loss"] < report["models"]["baseline"]["log_loss"]
    with pytest.warns(RuntimeWarning, match="did not converge"):
        model = mod.GoalModel(max_iter=1).fit(history(), "2021-01-01")
    assert model.to_dict()["fit_info"]["fallback"] == "baseline"
    assert min(model.predict_goals("strong", "weak")) > 0


def test_decay_and_neutral_symmetry():
    mod = module()
    data = history(100)
    short = mod.GoalModel(half_life_days=30).fit(data, "2020-04-11")
    long = mod.GoalModel(half_life_days=365).fit(data, "2020-04-11")
    assert short.to_dict()["weighted_matches"] < long.to_dict()["weighted_matches"]
    assert long.predict_goals("strong", "weak", True) == tuple(reversed(long.predict_goals("weak", "strong", True)))


def test_heldout_scores_do_not_affect_fitted_parameters(monkeypatch):
    mod = module()
    snapshots = []
    original = mod.GoalModel.fit
    def spy(self, data, cutoff):
        result = original(self, data, cutoff)
        snapshots.append(self.to_dict())
        return result
    monkeypatch.setattr(mod.GoalModel, "fit", spy)
    data = history(120)
    mod.evaluate(data, "2021-01-01")
    data.loc[96:, "home_goals"] = 25
    mod.evaluate(data, "2021-01-01")
    assert snapshots[0] == snapshots[1]


def test_defensive_strength_learned_from_goals_conceded():
    mod = module()
    data = history(600)
    data.loc[data.home == "strong", "away_goals"] = 0
    data.loc[data.away == "strong", "home_goals"] = 0
    model = mod.GoalModel().fit(data, "2022-01-01")
    assert model.predict_goals("average", "strong")[0] < model.predict_goals("average", "weak")[0]
    assert model.to_dict()["defence"]["strong"] > model.to_dict()["defence"]["weak"]


def test_timestamp_ties_do_not_cross_holdout_boundary():
    mod = module()
    data = history(100)
    data.loc[75:85, "date"] = data.loc[80, "date"]
    report = mod.evaluate(data, "2021-01-01")
    assert report["n_train"] == 75
    assert report["n_holdout"] == 25


def test_zero_goals_invalid_rows_and_input_immutability():
    mod = module()
    data = history(100)
    data[["home_goals", "away_goals"]] = 0.0
    data.loc[0, "home_goals"] = -1
    data.loc[1, "away_goals"] = .5
    data.loc[2, "away_goals"] = np.inf
    original = data.copy(deep=True)
    with pytest.warns(mod.DataWarning):
        model = mod.GoalModel().fit(data, "2021-01-01")
    assert model.n_matches == 97
    assert all(0 < x < 1 for x in model.predict_goals("strong", "weak"))
    pd.testing.assert_frame_equal(data, original)


def test_payload_copy_and_validation():
    mod = module()
    model = mod.GoalModel().fit(history(100), "2021-01-01")
    payload = model.to_dict()
    payload["attack"]["strong"] = 99
    assert model.to_dict()["attack"]["strong"] != 99
    payload["attack"]["strong"] = np.nan
    with pytest.raises(ValueError):
        mod.GoalModel.from_dict(payload)
    payload = model.to_dict()
    payload["schema_version"] = 999
    with pytest.raises(ValueError):
        mod.GoalModel.from_dict(payload)
