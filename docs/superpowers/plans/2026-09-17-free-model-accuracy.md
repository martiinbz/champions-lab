# Free Model Accuracy Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve weekly Champions forecasts using free data, leakage-safe dynamic Elo and league ratings, a Dixon-Coles score model, and chronologically validated ensemble selection.

**Architecture:** Keep ingestion, feature generation, match modelling, evaluation, simulation, and snapshot orchestration as separate units. Ratings are replayed chronologically and attached as pre-match features; both candidate models expose score probabilities, while a serializable ensemble exposes a common score-sampling interface to the tournament simulator.

**Tech Stack:** Python 3.11, pandas, NumPy, SciPy, scikit-learn metrics, pytest, PowerShell entrypoints.

---

## File map

- Create `src/champions/ratings.py`: competition classification, chronological Elo, league offsets, and leakage-safe feature frames.
- Create `src/champions/dixon_coles.py`: Dixon-Coles probability grid, fitted enhanced model, and serialization.
- Create `src/champions/ensemble.py`: weighted score-distribution ensemble and generic model deserialization.
- Modify `src/champions/data.py`: retain nullable free-source statistics and their provenance.
- Modify `src/champions/models.py`: add a normalized score-grid interface to the existing Poisson models and make evaluation consume it.
- Modify `src/champions/evaluation.py`: rolling candidate/ensemble evaluation and conservative promotion rule.
- Modify `src/champions/simulation.py`: sample all tournament scores through the common model interface.
- Modify `src/champions/pipeline.py`: fit the selected production model, persist diagnostics/features, and replay any model type.
- Modify `src/champions/cli.py`: expose explicit model choices without changing the default automatic workflow.
- Modify `README.md`, `docs/data-sources.md`, `docs/models.md`, `docs/validation.md`: explain data availability, commands, methodology, and limits.

### Task 1: Preserve optional free-source fields

**Files:**
- Modify: `src/champions/data.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Write failing contract and parser tests**

Add tests proving that Football-Data fields are retained, absent values stay null, and xG columns exist without fabricated values:

```python
def test_history_contract_has_nullable_optional_fields():
    assert mod.HISTORY_COLUMNS == [
        "date", "home", "away", "home_goals", "away_goals", "competition", "source",
        "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
        "odds_home", "odds_draw", "odds_away", "odds_provider", "odds_semantics",
        "home_xg", "away_xg",
    ]


def test_football_data_optional_values_are_retained_and_missing_is_null():
    csv = "Date,HomeTeam,AwayTeam,FTHG,FTAG,HS,AS,HST,AST,B365H,B365D,B365A\n01/09/2026,Arsenal,Chelsea,2,1,15,8,7,3,1.8,3.7,4.2\n"
    frame = mod._parse_football_data(csv, "E0", "https://example.test/E0.csv")
    row = frame.iloc[0]
    assert (row.home_shots, row.away_shots) == (15, 8)
    assert (row.odds_home, row.odds_draw, row.odds_away) == (1.8, 3.7, 4.2)
    assert row.odds_provider == "Bet365"
    assert row.odds_semantics == "football-data closing odds; research only"
    assert pd.isna(row.home_xg) and pd.isna(row.away_xg)
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `python -m pytest tests/test_data.py -q`

Expected: failure because the optional columns and `_parse_football_data` do not exist.

- [ ] **Step 3: Implement the normalized optional-field contract**

Add constants and one parser used by `fetch_history`:

```python
OPTIONAL_HISTORY_COLUMNS = [
    "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    "odds_home", "odds_draw", "odds_away", "odds_provider", "odds_semantics",
    "home_xg", "away_xg",
]
HISTORY_COLUMNS = [
    "date", "home", "away", "home_goals", "away_goals", "competition", "source",
    *OPTIONAL_HISTORY_COLUMNS,
]


def _parse_football_data(text: str, competition: str, source: str) -> pd.DataFrame:
    table = pd.read_csv(StringIO(text)).dropna(how="all")
    table = table.rename(columns={
        "HomeTeam": "home", "AwayTeam": "away", "FTHG": "home_goals", "FTAG": "away_goals",
        "HS": "home_shots", "AS": "away_shots", "HST": "home_shots_on_target",
        "AST": "away_shots_on_target", "B365H": "odds_home", "B365D": "odds_draw",
        "B365A": "odds_away",
    })
    table = table.loc[~table[["home_goals", "away_goals"]].isna().all(axis=1)].copy()
    table["date"] = pd.to_datetime(table["Date"], dayfirst=True, format="mixed", errors="raise").dt.strftime("%Y-%m-%dT23:59:59Z")
    table["home"] = table["home"].map(normalize_team)
    table["away"] = table["away"].map(normalize_team)
    table["competition"], table["source"] = competition, source
    for column in OPTIONAL_HISTORY_COLUMNS:
        if column not in table:
            table[column] = pd.NA
    has_bet365 = table["odds_home"].notna() & table["odds_draw"].notna() & table["odds_away"].notna()
    table["odds_provider"] = pd.Series(pd.NA, index=table.index, dtype="string").mask(has_bet365, "Bet365")
    table["odds_semantics"] = pd.Series(pd.NA, index=table.index, dtype="string").mask(
        has_bet365, "football-data closing odds; research only"
    )
    return table[HISTORY_COLUMNS]
```

For OpenFootball records, initialize every optional field with `pd.NA`. Update `validate_history` so only core columns are required to be non-null and numeric optional columns accept null but reject infinities and negative values. Do not expose odds or post-match statistics as modelling features.

- [ ] **Step 4: Run data tests and confirm GREEN**

Run: `python -m pytest tests/test_data.py -q`

Expected: all data tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/champions/data.py tests/test_data.py
git commit -m "feat: retain optional free match statistics"
```

### Task 2: Build chronological Elo and league-strength features

**Files:**
- Create: `src/champions/ratings.py`
- Create: `tests/test_ratings.py`

- [ ] **Step 1: Write failing chronology, league, and determinism tests**

```python
def test_future_result_cannot_change_earlier_features():
    base = matches()
    changed = base.copy()
    changed.loc[changed.index[-1], ["home_goals", "away_goals"]] = [9, 0]
    left = build_rating_features(base, "2026-10-01")
    right = build_rating_features(changed, "2026-10-01")
    pd.testing.assert_frame_equal(left.iloc[:-1], right.iloc[:-1])


def test_cross_league_win_moves_league_offsets_with_zero_weighted_mean():
    result = build_rating_features(matches_with_cross_league_games(), "2026-10-01")
    offsets = result.attrs["league_offsets"]
    assert offsets["E0"] > offsets["N1"]
    assert abs(sum(offsets.values())) < 1e-10


def test_rating_replay_is_input_order_independent():
    a = build_rating_features(matches(), "2026-10-01")
    b = build_rating_features(matches().sample(frac=1, random_state=7), "2026-10-01")
    pd.testing.assert_frame_equal(a, b)
```

- [ ] **Step 2: Run and confirm RED**

Run: `python -m pytest tests/test_ratings.py -q`

Expected: import failure for `champions.ratings`.

- [ ] **Step 3: Implement focused rating primitives**

Use explicit immutable parameters and pre-match rows:

```python
@dataclass(frozen=True)
class RatingConfig:
    initial: float = 1500.0
    k_factor: float = 24.0
    home_advantage: float = 55.0
    half_life_days: float = 365.0
    league_shrinkage: float = 20.0


def expected_score(home_rating: float, away_rating: float, home_advantage: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(home_rating + home_advantage - away_rating) / 400.0))


def goal_multiplier(goal_difference: int) -> float:
    return 1.0 if goal_difference <= 1 else 1.5 if goal_difference == 2 else (11.0 + goal_difference) / 8.0


def build_rating_features(history: pd.DataFrame, cutoff: str, config: RatingConfig = RatingConfig()) -> pd.DataFrame:
    """Return one row per completed match with ratings immediately before kickoff."""
```

Classify `UCL`, `UEL`, `UECL`, and their qualifiers as cross-league competitions; all others are domestic league IDs. Infer each club's league only from its latest strictly prior domestic appearance. Update league offsets only after a cross-league result, regularize offsets toward zero, and recenter to a zero weighted mean. Store final team ratings, league offsets, team-to-league mapping, and configuration in `DataFrame.attrs`.

- [ ] **Step 4: Add parameter-validation and sparse fallback tests**

Test invalid K/half-life, unseen teams receiving the population rating, and a disconnected league shrinking to zero rather than raising or receiving a reputation-based constant.

- [ ] **Step 5: Run rating tests and confirm GREEN**

Run: `python -m pytest tests/test_ratings.py -q`

Expected: all rating tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/champions/ratings.py tests/test_ratings.py
git commit -m "feat: add leakage-safe dynamic ratings"
```

### Task 3: Give every model a coherent score-distribution interface

**Files:**
- Modify: `src/champions/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write failing score-grid and seeded-sampling tests**

```python
def test_poisson_score_probabilities_are_normalized_and_match_outcomes():
    model = mod.GoalModel().fit(history(), "2022-01-01")
    grid = model.score_probabilities("A", "B", max_goals=12)
    assert grid.shape == (13, 13)
    assert grid.sum() == pytest.approx(1.0)
    outcomes = model.predict_outcomes("A", "B")
    assert np.tril(grid, -1).sum() == pytest.approx(outcomes["home"], abs=2e-4)


def test_sample_scores_is_deterministic_for_seed():
    model = mod.GoalModel().fit(history(), "2022-01-01")
    assert np.array_equal(
        model.sample_scores("A", "B", 100, np.random.default_rng(3)),
        model.sample_scores("A", "B", 100, np.random.default_rng(3)),
    )
```

- [ ] **Step 2: Run and confirm RED**

Run: `python -m pytest tests/test_models.py -q`

Expected: missing `score_probabilities` and `sample_scores`.

- [ ] **Step 3: Implement reusable normalized grid and sampling**

```python
def independent_poisson_grid(home_rate: float, away_rate: float, max_goals: int = 12) -> np.ndarray:
    goals = np.arange(max_goals + 1)
    home = poisson.pmf(goals, home_rate)
    away = poisson.pmf(goals, away_rate)
    grid = np.outer(home, away)
    grid[-1, :] += (1.0 - home.sum()) * away
    grid[:, -1] += home * (1.0 - away.sum())
    grid[-1, -1] += (1.0 - home.sum()) * (1.0 - away.sum())
    return grid / grid.sum()
```

Add `score_probabilities` and `sample_scores` to `BaselineModel`, with flat-CDF sampling returning an integer array shaped `(size, 2)`. Refactor `_metrics` to derive 1X2 and observed-score likelihood from the score grid while retaining expected goals for goal-loss reporting.

- [ ] **Step 4: Run model tests and confirm GREEN**

Run: `python -m pytest tests/test_models.py -q`

Expected: all model tests pass and old serialization remains compatible.

- [ ] **Step 5: Commit**

```powershell
git add src/champions/models.py tests/test_models.py
git commit -m "feat: expose coherent score distributions"
```

### Task 4: Implement enhanced Dixon-Coles model

**Files:**
- Create: `src/champions/dixon_coles.py`
- Create: `tests/test_dixon_coles.py`

- [ ] **Step 1: Write failing low-score, feature, leakage, and round-trip tests**

```python
def test_dixon_coles_changes_only_low_score_cells_before_normalization():
    base = independent_poisson_grid(1.4, 1.1, 12)
    adjusted = dixon_coles_grid(1.4, 1.1, rho=-0.08, max_goals=12)
    assert adjusted.sum() == pytest.approx(1.0)
    assert adjusted[0, 0] != pytest.approx(base[0, 0])
    assert adjusted[1, 1] != pytest.approx(base[1, 1])
    assert adjusted[3, 2] / adjusted[2, 3] == pytest.approx(base[3, 2] / base[2, 3])


def test_model_roundtrip_preserves_grid():
    fitted = DixonColesModel(max_iter=100).fit(history(), "2026-09-01")
    restored = DixonColesModel.from_dict(fitted.to_dict())
    np.testing.assert_allclose(
        fitted.score_probabilities("Arsenal", "Chelsea"),
        restored.score_probabilities("Arsenal", "Chelsea"),
    )
```

- [ ] **Step 2: Run and confirm RED**

Run: `python -m pytest tests/test_dixon_coles.py -q`

Expected: import failure for `champions.dixon_coles`.

- [ ] **Step 3: Implement the correlation function and model API**

```python
def tau(home_goals: int, away_goals: int, home_rate: float, away_rate: float, rho: float) -> float:
    if (home_goals, away_goals) == (0, 0): return 1.0 - home_rate * away_rate * rho
    if (home_goals, away_goals) == (0, 1): return 1.0 + home_rate * rho
    if (home_goals, away_goals) == (1, 0): return 1.0 + away_rate * rho
    if (home_goals, away_goals) == (1, 1): return 1.0 - rho
    return 1.0


def dixon_coles_grid(home_rate: float, away_rate: float, rho: float, max_goals: int = 12) -> np.ndarray:
    grid = independent_poisson_grid(home_rate, away_rate, max_goals)
    for home_goals, away_goals in ((0, 0), (0, 1), (1, 0), (1, 1)):
        grid[home_goals, away_goals] *= tau(home_goals, away_goals, home_rate, away_rate, rho)
    if not np.isfinite(grid).all() or (grid < 0).any() or grid.sum() <= 0:
        raise ValueError("invalid Dixon-Coles correction")
    return grid / grid.sum()


def _linear_rates(x, home_index, away_index, elo_difference, home_league, away_league, neutral):
    intercept, home_advantage, elo_scale, league_scale = x[:4]
    n_teams = (len(x) - 5) // 2
    attack, defence = x[4:4+n_teams], x[4+n_teams:4+2*n_teams]
    venue = 0.0 if neutral else home_advantage
    home_log = intercept + venue + attack[home_index] - defence[away_index]
    away_log = intercept + attack[away_index] - defence[home_index]
    adjustment = elo_scale * elo_difference + league_scale * (home_league - away_league)
    return np.exp(np.clip([home_log + adjustment, away_log - adjustment], -8, 5))
```

Implement `DixonColesModel` as a `BaselineModel` subclass with constructor parameters `half_life_days=180.0`, `regularization=8.0`, `elo_scale=0.0015`, `league_scale=0.0015`, `max_iter=500`, and `tolerance=1e-9`. Its public methods are `fit`, `score_probabilities`, `sample_scores`, `to_dict`, and `from_dict`, matching the argument order used by `GoalModel` and Task 3. The objective is weighted negative log likelihood of observed score cells plus L2 penalties. Log rates use `_linear_rates`; the final optimizer coordinate is `rho`, bounded to `[-0.2, 0.2]`, and any candidate with a nonpositive observed low-score correction returns infinite loss. If optimization fails, record the reason and copy fitted independent-Poisson parameters rather than partial optimizer output. Serialization schema version 1 stores constructor parameters, rating configuration, final ratings and league offsets, coverage, fitted coefficients, `rho`, and `fit_info`; restoration rejects missing keys, nonfinite values, negative coverage, or weights with inconsistent team keys.

- [ ] **Step 4: Add optimizer-failure and unknown-team fallback tests**

Monkeypatch `minimize` to fail and assert finite probabilities plus `fit_info["fallback"] == "poisson"`. Assert unseen teams use population attack/defence, final Elo, and zero league offset with a recorded coverage warning.

- [ ] **Step 5: Run focused tests and confirm GREEN**

Run: `python -m pytest tests/test_dixon_coles.py tests/test_ratings.py tests/test_models.py -q`

Expected: all focused tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/champions/dixon_coles.py tests/test_dixon_coles.py
git commit -m "feat: add Elo-aware Dixon-Coles model"
```

### Task 5: Select and serialize a chronological ensemble

**Files:**
- Create: `src/champions/ensemble.py`
- Modify: `src/champions/evaluation.py`
- Modify: `tests/test_evaluation.py`
- Create: `tests/test_ensemble.py`

- [ ] **Step 1: Write failing mixture and promotion-rule tests**

```python
def test_ensemble_grid_is_weighted_component_distribution():
    ensemble = EnsembleModel([FixedModel(grid_a), FixedModel(grid_b)], [0.25, 0.75])
    np.testing.assert_allclose(ensemble.score_probabilities("A", "B"), 0.25 * grid_a + 0.75 * grid_b)


def test_auto_requires_two_selection_folds_and_minimum_improvement(monkeypatch):
    report = evaluate_rolling(history(), "2026-09-01", folds=3)
    assert report["promotion"]["minimum_log_loss_improvement"] == 0.005
    assert report["selection_folds"] == 2
    assert report["assessment_fold"] == 3


def test_final_block_cannot_change_candidate_or_weight(monkeypatch):
    first = evaluate_rolling(history(), "2026-09-01", folds=3)
    changed = mutate_only_final_assessment_scores(history())
    second = evaluate_rolling(changed, "2026-09-01", folds=3)
    assert first["selected_model"] == second["selected_model"]
    assert first["ensemble_weights"] == second["ensemble_weights"]
```

- [ ] **Step 2: Run and confirm RED**

Run: `python -m pytest tests/test_ensemble.py tests/test_evaluation.py -q`

Expected: ensemble import failure and missing enhanced candidates.

- [ ] **Step 3: Implement distribution mixing and generic restoration**

```python
MODEL_LOADERS = {
    "GoalModel": GoalModel.from_dict,
    "BaselineModel": BaselineModel.from_dict,
    "DixonColesModel": DixonColesModel.from_dict,
}


class EnsembleModel:
    def __init__(self, components, weights):
        weights = np.asarray(weights, dtype=float)
        if len(components) != len(weights) or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
            raise ValueError("ensemble weights must be nonnegative and sum to one")
        self.components, self.weights = list(components), weights

    def score_probabilities(self, home, away, neutral=False, max_goals=12):
        return sum(w * m.score_probabilities(home, away, neutral, max_goals)
                   for m, w in zip(self.components, self.weights))

    def sample_scores(self, home, away, size, rng, neutral=False):
        grid = self.score_probabilities(home, away, neutral)
        flat = rng.choice(grid.size, size=size, p=grid.ravel())
        return np.column_stack(np.unravel_index(flat, grid.shape))
```

Add `to_dict`, `from_dict`, and `model_from_dict` with deep-copy and numeric validation.

- [ ] **Step 4: Extend rolling evaluation**

For each selection fold, fit Poisson and Dixon-Coles on strictly prior matches and cache their UCL score grids. Search ensemble Poisson weight over `np.linspace(0, 1, 21)` and calculate log loss, Brier score, goal log loss, and ECE from the mixed grids. Aggregate by holdout match count. Promote Dixon-Coles or ensemble only when its selection log loss is at least `0.005` below Poisson across at least two selection folds; otherwise choose Poisson. Fit the already-selected definition on the assessment train block and report assessment metrics without reconsidering selection.

- [ ] **Step 5: Run evaluation and ensemble tests and confirm GREEN**

Run: `python -m pytest tests/test_ensemble.py tests/test_evaluation.py -q`

Expected: all tests pass, including the existing untouched-final-block test.

- [ ] **Step 6: Commit**

```powershell
git add src/champions/ensemble.py src/champions/evaluation.py tests/test_ensemble.py tests/test_evaluation.py
git commit -m "feat: validate and select probabilistic ensemble"
```

### Task 6: Use coherent distributions throughout tournament simulation

**Files:**
- Modify: `src/champions/simulation.py`
- Modify: `src/champions/knockout.py`
- Modify: `tests/test_simulation.py`
- Modify: `tests/test_knockout.py`

- [ ] **Step 1: Write failing interface-use and reproducibility tests**

```python
class DistributionOnlyModel:
    def score_probabilities(self, home, away, neutral=False, max_goals=12):
        grid = np.zeros((13, 13))
        grid[2 if not neutral else 1, 0] = 1.0
        return grid


def test_simulation_accepts_distribution_only_model_and_is_reproducible():
    one, _ = simulate(schedule(), DistributionOnlyModel(), simulations=1200, seed=9)
    two, _ = simulate(schedule(), DistributionOnlyModel(), simulations=1200, seed=9)
    pd.testing.assert_frame_equal(one, two)
```

- [ ] **Step 2: Run and confirm RED**

Run: `python -m pytest tests/test_simulation.py tests/test_knockout.py -q`

Expected: `predict_goals` is still required.

- [ ] **Step 3: Add one cached score sampler for league and knockout**

```python
def _score_grid(model, home: str, away: str, neutral: bool) -> np.ndarray:
    if hasattr(model, "score_probabilities"):
        grid = np.asarray(model.score_probabilities(home, away, neutral=neutral), dtype=float)
    else:
        grid = independent_poisson_grid(*model.predict_goals(home, away, neutral=neutral))
    if grid.ndim != 2 or not np.isfinite(grid).all() or (grid < 0).any() or not np.isclose(grid.sum(), 1):
        raise ValueError(f"Invalid score distribution for {home} vs {away}")
    return grid / grid.sum()


def _draw_grid(grid: np.ndarray, size: int, rng) -> tuple[np.ndarray, np.ndarray]:
    flat = rng.choice(grid.size, size=size, p=grid.ravel())
    return np.unravel_index(flat, grid.shape)
```

Cache home and neutral grids for every ordered pair. Use `_draw_grid` for league games, both knockout legs, finals, and extra time (extra time remains an explicit one-third-rate approximation derived from each model's expected goals and is disclosed). Update diagnostics to identify `score_distribution_sampling` and the selected model type instead of claiming all scores are independent Poisson.

- [ ] **Step 4: Run simulation tests and confirm GREEN**

Run: `python -m pytest tests/test_simulation.py tests/test_knockout.py -q`

Expected: all conservation, bracket, determinism, and distribution-interface tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/champions/simulation.py src/champions/knockout.py tests/test_simulation.py tests/test_knockout.py
git commit -m "feat: simulate from coherent score distributions"
```

### Task 7: Integrate automatic selection into snapshots and replay

**Files:**
- Modify: `src/champions/pipeline.py`
- Modify: `src/champions/cli.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_entrypoints.py`

- [ ] **Step 1: Write failing production-model and replay tests**

```python
def test_auto_fits_selected_definition_and_persists_rating_metadata(prepared_inputs, monkeypatch):
    target = run(prepared_inputs, "2026-09-17T00:00:00Z", simulations=100, model_kind="auto")
    model = json.loads((target / "model.json").read_text(encoding="utf-8"))
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert model["model_type"] in {"GoalModel", "DixonColesModel", "EnsembleModel"}
    assert metadata["model_selection"]["selection_folds"] >= 2 or metadata["model"] == "poisson"
    assert "rating_config" in metadata["model_fit_info"]
    assert reproduce(prepared_inputs, metadata["run_id"])["probabilities_reproduced"]
```

- [ ] **Step 2: Run and confirm RED**

Run: `python -m pytest tests/test_pipeline.py tests/test_entrypoints.py -q`

Expected: auto only understands baseline/Poisson and replay only loads `GoalModel`.

- [ ] **Step 3: Centralize fitting from the evaluation decision**

Add:

```python
def fit_production_model(history: pd.DataFrame, cutoff: str, decision: dict):
    selected = decision.get("selected_model", "poisson")
    if selected == "poisson":
        return GoalModel().fit(history, cutoff)
    if selected == "dixon_coles":
        return DixonColesModel().fit(history, cutoff)
    if selected == "ensemble":
        components = [GoalModel().fit(history, cutoff), DixonColesModel().fit(history, cutoff)]
        return EnsembleModel(components, decision["ensemble_weights"])
    raise ValueError(f"Modelo seleccionado desconocido: {selected}")
```

Use it in `run`; use `model_from_dict` in `reproduce`. Persist selection fold scores, chosen weights, promotion threshold, final assessment, rating config, source-generation hash, and failures/fallbacks. Add CLI choices `dixon-coles` and `ensemble`; explicit ensemble uses the latest validated weights and errors if no valid weights exist. Preserve `auto` as default.

- [ ] **Step 4: Confirm update workflow still uses the requested simulation count**

Add an entrypoint test asserting `Actualizar-jornada.ps1 -Simulaciones 200000` forwards `200000` after updating data and does not hide an update failure.

- [ ] **Step 5: Run pipeline tests and confirm GREEN**

Run: `python -m pytest tests/test_pipeline.py tests/test_entrypoints.py -q`

Expected: all pipeline, replay, and PowerShell contract tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/champions/pipeline.py src/champions/cli.py tests/test_pipeline.py tests/test_entrypoints.py
git commit -m "feat: deploy validated model selection in weekly runs"
```

### Task 8: Document methodology and run full verification

**Files:**
- Modify: `README.md`
- Modify: `docs/data-sources.md`
- Modify: `docs/models.md`
- Modify: `docs/validation.md`

- [ ] **Step 1: Update user-facing documentation**

Document these exact commands:

```powershell
# Download current free data, revalidate models, and run 200,000 simulations
.\Actualizar-jornada.ps1 -Simulaciones 200000

# Re-simulate the current local data without downloading
.\Nueva-simulacion.ps1 -Simulaciones 200000 -Semilla 42
```

Explain that UEFA, Football-Data.co.uk, and OpenFootball are the current sources; xG columns are reserved but null; closing odds and post-match shot statistics are stored for research and excluded from pre-match training; ratings are strictly pre-match; model promotion uses selection folds only; the newest assessment block is never used to tune or reverse selection. Record remaining limitations: no current reliable free xG feed, injuries/lineups, exact disciplinary/coefficient tie-breakers, changing future strength, or learned penalty probabilities.

- [ ] **Step 2: Run formatting and placeholder checks**

Run:

```powershell
git diff --check
rg -n "TBD|TODO|implement later|fill in" README.md docs src tests
```

Expected: `git diff --check` exits zero and the placeholder search returns no newly introduced placeholders.

- [ ] **Step 3: Run the entire test suite**

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 4: Exercise one deterministic smoke simulation**

Run:

```powershell
python -m champions.cli simulate --simulations 100 --seed 17 --model auto
python -m champions.cli list
```

Expected: a new immutable snapshot is printed; its metadata identifies the selected model, selection evidence, data generation, ratings configuration, and 100 simulations.

- [ ] **Step 5: Reproduce the smoke snapshot**

Run:

```powershell
$runId = (Get-ChildItem results\snapshots -Directory | Sort-Object Name | Select-Object -Last 1).Name
python -m champions.cli reproduce $runId
```

Expected: `hashes_valid` and `probabilities_reproduced` are both `true`.

- [ ] **Step 6: Commit documentation**

```powershell
git add README.md docs/data-sources.md docs/models.md docs/validation.md
git commit -m "docs: explain free-data accuracy pipeline"
```

- [ ] **Step 7: Inspect final repository state**

Run: `git status --short && git log --oneline -10`

Expected: clean working tree and one focused commit per completed task.
