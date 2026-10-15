# Temporal Backtesting and Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add leakage-safe Champions matchday backtesting and promote a serializable 1X2 calibrator only when it improves future, unseen predictions.

**Architecture:** `calibration.py` owns a regularized temperature-plus-class-offset transformation and a score-grid wrapper. `evaluation.py` produces chronological pre-match predictions for each UCL window, keeps the final window untouched, and makes the calibration decision. `pipeline.py` fits the selected base model and conditionally wraps it; the simulator samples its score grids in league and knockout normal time.

**Tech Stack:** Python 3.13, pandas, NumPy, SciPy, pytest.

---

## Files and responsibilities

- Create `src/champions/calibration.py`: validated outcome vectors, fitting/serialization of `OutcomeCalibrator`, and `CalibratedModel` score-grid wrapper.
- Modify `src/champions/ensemble.py`: restore the calibrated wrapper recursively from immutable snapshot JSON.
- Modify `src/champions/evaluation.py`: build strictly chronological UCL windows and return the base-vs-calibrated promotion decision.
- Modify `src/champions/simulation.py`: sample model score distributions for knockout normal-time legs as it already does in the league phase.
- Modify `src/champions/pipeline.py`: conditionally wrap the production model and expose a durable `backtest` report writer.
- Modify `src/champions/cli.py`: add `champions backtest` without changing existing simulation commands.
- Create `tests/test_calibration.py`: unit and serialization tests for the calibrator and wrapper.
- Modify `tests/test_evaluation.py`, `tests/test_simulation.py`, `tests/test_pipeline.py`, and `tests/test_ensemble.py`: temporal-leakage, promotion, knockout, CLI/persistence, and restoration coverage.
- Modify `docs/models.md` and `README.md`: document what is calibrated, the out-of-sample guard rail, and the `backtest` command.

### Task 1: Add a validated, serializable score-distribution calibrator

**Files:**
- Create: `src/champions/calibration.py`
- Create: `tests/test_calibration.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest

from champions.calibration import OutcomeCalibrator, CalibratedModel


class GridModel:
    def __init__(self, grid):
        self.grid = np.asarray(grid, dtype=float)
        self.warnings = []
        self.fit_info = {"method": "test"}

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
```

- [ ] **Step 2: Run the focused test to verify RED**

Run: `python -m pytest tests/test_calibration.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'champions.calibration'`.

- [ ] **Step 3: Implement the minimal public API**

```python
OUTCOMES = ("home", "draw", "away")


def grid_outcomes(grid: np.ndarray) -> np.ndarray:
    values = np.asarray(grid, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1.0):
        raise ValueError("score grid must be finite, nonnegative and normalized")
    return np.array([np.tril(values, -1).sum(), np.trace(values), np.triu(values, 1).sum()])


class OutcomeCalibrator:
    def transform(self, probabilities) -> dict[str, float]:
        base = _normalized_vector(probabilities)
        values = _softmax(np.log(base) / np.exp(self.log_temperature) + self.offsets)
        return dict(zip(OUTCOMES, map(float, values)))


class CalibratedModel:
    def score_probabilities(self, home, away, neutral=False, max_goals=12) -> np.ndarray:
        base = np.asarray(self.base_model.score_probabilities(home, away, neutral=neutral, max_goals=max_goals), dtype=float)
        before = grid_outcomes(base)
        after = np.asarray(list(self.calibrator.transform(before).values()))
        multiplier = np.divide(after, before, out=np.zeros(3), where=before > 0)
        result = base.copy()
        result[np.tril_indices_from(result, -1)] *= multiplier[0]
        result[np.diag_indices_from(result)] *= multiplier[1]
        result[np.triu_indices_from(result, 1)] *= multiplier[2]
        return result / result.sum()
```

Implement constructor validation, `predict_goals` delegation, and schema-version-1 `to_dict`/ `from_dict`. The wrapper must preserve `warnings`, `coverage`, `params`, and `fit_info` from its base model.

- [ ] **Step 4: Run the focused test to verify GREEN**

Run: `python -m pytest tests/test_calibration.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/champions/calibration.py tests/test_calibration.py
git commit -m "feat: add serializable outcome calibrator"
```

### Task 2: Fit the calibrator and restore it from immutable snapshots

**Files:**
- Modify: `src/champions/calibration.py`
- Modify: `src/champions/ensemble.py`
- Modify: `tests/test_calibration.py`
- Modify: `tests/test_ensemble.py`

- [ ] **Step 1: Write failing fit and restoration tests**

```python
def test_fit_reduces_log_loss_on_a_deliberately_overconfident_training_sample():
    probabilities = np.tile([.94, .03, .03], (90, 1))
    outcomes = np.array([0] * 54 + [1] * 18 + [2] * 18)
    calibrator = OutcomeCalibrator.fit(probabilities, outcomes, l2=1.0, min_samples=60)
    raw_loss = OutcomeCalibrator.log_loss(probabilities, outcomes)
    fitted = np.array([list(calibrator.transform(row).values()) for row in probabilities])
    assert OutcomeCalibrator.log_loss(fitted, outcomes) < raw_loss


def test_calibrated_goal_model_json_round_trip_keeps_score_distribution():
    base = GoalModel().fit(history, "2026-01-01")
    model = CalibratedModel(base, OutcomeCalibrator(.1, [.05, -.1, .05]))
    restored = model_from_dict(json.loads(json.dumps(model.to_dict())))
    np.testing.assert_allclose(restored.score_probabilities("A", "B"), model.score_probabilities("A", "B"))
```

- [ ] **Step 2: Run the focused tests to verify RED**

Run: `python -m pytest tests/test_calibration.py tests/test_ensemble.py -q`

Expected: failures because `OutcomeCalibrator.fit` and calibrated model restoration do not exist.

- [ ] **Step 3: Implement regularized fitting and recursive restoration**

```python
@classmethod
def fit(cls, probabilities, outcomes, *, l2: float = 1.0, min_samples: int = 60):
    matrix = _normalized_matrix(probabilities)
    target = _outcome_indices(outcomes, len(matrix))
    if len(matrix) < min_samples or len(np.unique(target)) < 3:
        raise ValueError("calibration requires enough observations of home, draw and away outcomes")
    def objective(values):
        candidate = cls(values[0], [values[1], values[2], -values[1] - values[2]])
        return cls.log_loss(candidate.transform_matrix(matrix), target) + l2 * float(np.dot(values, values)) / len(matrix)
    result = minimize(objective, np.zeros(3), method="L-BFGS-B", bounds=[(-1.0, 1.0), (-2.0, 2.0), (-2.0, 2.0)])
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError("calibration optimization did not converge")
    return cls(result.x[0], [result.x[1], result.x[2], -result.x[1] - result.x[2]])
```

Add `CalibratedModel.to_dict` with `model_type: "CalibratedModel"`, `schema_version: 1`, `base_model`, and `calibrator`. Extend `model_from_dict` to restore the wrapper recursively.

- [ ] **Step 4: Run the focused tests to verify GREEN**

Run: `python -m pytest tests/test_calibration.py tests/test_ensemble.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/champions/calibration.py src/champions/ensemble.py tests/test_calibration.py tests/test_ensemble.py
git commit -m "feat: fit and restore calibrated score models"
```

### Task 3: Backtest temporal UCL windows and gate calibration

**Files:**
- Modify: `src/champions/evaluation.py`
- Modify: `tests/test_evaluation.py`

- [ ] **Step 1: Write failing temporal-isolation and promotion tests**

```python
def test_ucl_windows_use_only_prior_matches_and_preserve_final_assessment():
    frame = synthetic_ucl_history(days=720)
    report = evaluate_rolling(frame, "2026-01-01", folds=4)
    assert report["status"] == "ok"
    assert len(report["folds"]) == 4
    assert report["folds"][-1]["purpose"] == "assessment"
    assert all(fold["train_last_date"] < fold["holdout_start"] for fold in report["folds"])
    altered = frame.copy()
    altered.loc[altered.date >= pd.Timestamp(report["folds"][-1]["holdout_start"]), ["home_goals", "away_goals"]] = [8, 0]
    changed = evaluate_rolling(altered, "2026-01-01", folds=4)
    assert report["calibration"]["promoted"] == changed["calibration"]["promoted"]
    assert report["calibration"]["selection"] == changed["calibration"]["selection"]


def test_calibration_is_not_promoted_when_selection_loss_does_not_improve(monkeypatch):
    monkeypatch.setattr("champions.evaluation.OutcomeCalibrator.fit", lambda *args, **kwargs: IdentityCalibrator())
    report = evaluate_rolling(synthetic_ucl_history(days=720), "2026-01-01", folds=4)
    assert report["calibration"]["promoted"] is False
    assert report["calibration"]["reason"] == "insufficient_selection_improvement"
```

- [ ] **Step 2: Run the evaluation tests to verify RED**

Run: `python -m pytest tests/test_evaluation.py -q`

Expected: failures because the current report has no calibration decision.

- [ ] **Step 3: Refactor evaluation into explicit chronological prediction records**

Add `_prediction_records(model, heldout)`, returning a normalized Nx3 matrix from `grid_outcomes(model.score_probabilities(...))` plus target result indices. Change the default to four temporal folds: folds 1–2 create out-of-fold records to fit the calibrator; fold 3 decides promotion; fold 4 is assessment only. Continue fitting each candidate only with rows before that fold start.

Return this stable decision shape:

```python
"calibration": {
    "status": "ok" | "insufficient_data",
    "promoted": bool,
    "reason": "improved_selection" | "insufficient_selection_improvement" | "insufficient_class_coverage" | "optimizer_failed",
    "minimum_log_loss_improvement": 0.005,
    "selection": {"n_train": int, "n_validate": int, "base": {"log_loss": float, "brier": float}, "calibrated": {"log_loss": float, "brier": float}},
    "assessment": {"base": {"log_loss": float, "brier": float}, "calibrated": {"log_loss": float, "brier": float}},
    "payload": dict | None,
}
```

Promote only when calibration improves fold-3 weighted log-loss by at least .005 and does not worsen Brier. The final fold must never affect `payload` or `promoted`. Preserve current base-model selection report fields.

- [ ] **Step 4: Run evaluation tests to verify GREEN**

Run: `python -m pytest tests/test_evaluation.py tests/test_models.py -q`

Expected: PASS; final-window alteration changes assessment metrics only.

- [ ] **Step 5: Commit**

```powershell
git add src/champions/evaluation.py tests/test_evaluation.py
git commit -m "feat: backtest UCL windows and gate calibration"
```

### Task 4: Use score distributions in knockout normal time

**Files:**
- Modify: `src/champions/simulation.py`
- Modify: `tests/test_simulation.py`
- Modify: `tests/test_knockout.py`

- [ ] **Step 1: Write a failing knockout-grid test**

```python
def test_knockout_normal_time_uses_available_score_grids(monkeypatch):
    fixture = schedule()
    model = DrawDistributionModel()
    calls = []
    monkeypatch.setattr(simulation, "_sample_grid_pairs", lambda first, second, grids, neutral, rng: (calls.append(True) or (np.zeros(first.shape, dtype=int), np.zeros(second.shape, dtype=int))))
    simulation.simulate(fixture, model, simulations=32, seed=4)
    assert calls
```

- [ ] **Step 2: Run the focused test to verify RED**

Run: `python -m pytest tests/test_simulation.py::test_knockout_normal_time_uses_available_score_grids -q`

Expected: failure because `_sample_grid_pairs` does not exist and knockout code calls `rng.poisson`.

- [ ] **Step 3: Implement batched normal-time grid sampling**

```python
def _sample_grid_pairs(first, second, grids, neutral, rng):
    home, away = np.empty(first.shape, dtype=np.int64), np.empty(second.shape, dtype=np.int64)
    for h, a in np.unique(np.column_stack([first.ravel(), second.ravel()]), axis=0):
        mask = (first == h) & (second == a)
        grid = grids[int(neutral), h, a]
        draw = rng.choice(grid.size, size=int(mask.sum()), p=grid.ravel())
        home[mask], away[mask] = np.unravel_index(draw, grid.shape)
    return home, away
```

Pass optional `score_grids` through `_two_leg`, `_final`, and `_bracket`. Use it for normal-time legs; retain rate-based one-third Poisson extra time and the rate fallback for models without score distributions. Add a `knockout_normal_time_sampling` diagnostics value.

- [ ] **Step 4: Run simulation tests to verify GREEN**

Run: `python -m pytest tests/test_simulation.py tests/test_knockout.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/champions/simulation.py tests/test_simulation.py tests/test_knockout.py
git commit -m "feat: use score distributions in knockout normal time"
```

### Task 5: Promote calibrated models in the pipeline and persist backtests

**Files:**
- Modify: `src/champions/pipeline.py`
- Modify: `src/champions/cli.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_entrypoints.py`

- [ ] **Step 1: Write failing pipeline and CLI tests**

```python
def test_fit_production_model_wraps_only_a_promoted_calibration():
    plain = fit_production_model(minimal_history(), "2026-01-01", {"selected_model": "poisson", "calibration": {"promoted": False}})
    payload = OutcomeCalibrator().to_dict()
    calibrated = fit_production_model(minimal_history(), "2026-01-01", {"selected_model": "poisson", "calibration": {"promoted": True, "payload": payload}})
    assert type(plain).__name__ == "GoalModel"
    assert type(calibrated).__name__ == "CalibratedModel"


def test_backtest_persists_evaluation_report(tmp_path):
    prepared_inputs(tmp_path)
    path = backtest(tmp_path, "2026-09-16")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["cutoff"] == "2026-09-16T00:00:00+00:00"
    assert "ucl_rolling" in report["evaluation"]
```

- [ ] **Step 2: Run focused tests to verify RED**

Run: `python -m pytest tests/test_pipeline.py tests/test_entrypoints.py -q`

Expected: failures because pipeline ignores calibration and does not expose `backtest`.

- [ ] **Step 3: Integrate calibration and a durable report writer**

```python
def fit_production_model(history, cutoff, decision):
    base = _fit_base_model(history, cutoff, decision)
    calibration = decision.get("calibration", {})
    if calibration.get("promoted"):
        return CalibratedModel(base, OutcomeCalibrator.from_dict(calibration["payload"]))
    return base


def backtest(root: Path, cutoff: str) -> Path:
    history = _load_verified_history(root)
    evaluation = evaluate_models(history, cutoff)
    target = root / "results" / "backtests" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, {"cutoff": utc(cutoff).isoformat(), "created_at": datetime.now(timezone.utc).isoformat(), "evaluation": evaluation})
    return target
```

Extract verified processed-history loading shared by `run` and `backtest`. Pass `evaluation.get("ucl_rolling", {}).get("calibration")` to production fitting. Add the `backtest` CLI parser and return a JSON path under the key `backtest`. Keep explicit `--model` choices valid; calibration remains conditional on evidence.

- [ ] **Step 4: Run focused tests to verify GREEN**

Run: `python -m pytest tests/test_pipeline.py tests/test_entrypoints.py -q`

Expected: PASS, including snapshot reproduction when a wrapper is promoted.

- [ ] **Step 5: Commit**

```powershell
git add src/champions/pipeline.py src/champions/cli.py tests/test_pipeline.py tests/test_entrypoints.py
git commit -m "feat: promote calibrated models from temporal backtests"
```

### Task 6: Document and verify the integrated feature

**Files:**
- Modify: `README.md`
- Modify: `docs/models.md`

- [ ] **Step 1: Add user-facing usage and limits**

Add:

```powershell
python -m champions.cli --root . backtest --cutoff 2026-09-18T08:26:04Z
```

Document that reports go to `results/backtests/`, promotion needs at least .005 unseen log-loss improvement without worse Brier, and extra time remains rate-Poisson based.

- [ ] **Step 2: Run targeted verification**

Run: `python -m pytest tests/test_calibration.py tests/test_evaluation.py tests/test_simulation.py tests/test_knockout.py tests/test_pipeline.py tests/test_ensemble.py tests/test_entrypoints.py -q`

Expected: PASS.

- [ ] **Step 3: Run full verification**

Run: `python -m pytest -q; git diff --check`

Expected: all tests pass and no whitespace errors. Existing sparse-coverage and UEFA-tiebreaker warnings may remain when their documented test fixtures trigger them.

- [ ] **Step 4: Commit**

```powershell
git add README.md docs/models.md
git commit -m "docs: explain calibrated temporal backtests"
git status --short
```

Expected: no tracked uncommitted changes.

