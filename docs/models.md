# Goal models

`champions.models` depends only on NumPy, pandas, SciPy and the standard library.
It performs no downloads. The default is a regularized, time-decayed independent
Poisson model; a smoothed league baseline is available through the same API.

## Pipeline interface

```python
from champions.models import GoalModel, evaluate

model = GoalModel().fit(history, cutoff="2026-09-16")
home_rate, away_rate = model.predict_goals("Arsenal", "Barcelona", neutral=False)
probabilities = model.predict_outcomes("Arsenal", "Barcelona")
# {'home': float, 'draw': float, 'away': float}; sums to one.

payload = model.to_dict()                  # json.dumps(payload, allow_nan=False)
restored = GoalModel.from_dict(payload)    # no fitting or external data needed
report = evaluate(history, "2026-09-16")

# Explicit optional selection. Evaluation itself never switches or refits model.
selected = GoalModel(kind=report["selected_model"]).fit(history, "2026-09-16")
baseline = GoalModel(kind="baseline").fit(history, "2026-09-16")
coverage_messages = model.warnings         # list[str], also in payload['warnings']
coverage = payload["coverage"]             # team -> matches, weighted_matches
```

Exact signatures:

```python
GoalModel(*, kind="poisson", half_life_days=365.0, regularization=5.0,
          prior_goals=1.35, prior_home_advantage=0.15, prior_matches=10.0,
          min_team_matches=5, max_iter=500, tolerance=1e-9)
GoalModel.fit(history: pandas.DataFrame, cutoff: str) -> GoalModel
GoalModel.predict_goals(home: str, away: str, neutral: bool = False) -> tuple[float, float]
GoalModel.predict_outcomes(home: str, away: str, neutral: bool = False) -> dict[str, float]
GoalModel.to_dict() -> dict
GoalModel.from_dict(payload: dict) -> GoalModel  # classmethod
evaluate(history: pandas.DataFrame, cutoff: str, *, holdout_fraction=0.2,
         min_train=30, min_holdout=10, max_holdout=2000,
         calibration_bins=10, model_params=None) -> dict
outcome_probabilities(home_goals: float, away_goals: float) -> dict[str, float]
```

`BaselineModel` is also exported; it accepts the common decay/prior/coverage
arguments and has the same fit/predict/serialization methods. Restore its payload
with `BaselineModel.from_dict`. `GoalModel(kind="baseline")` always round-trips
through `GoalModel.from_dict`. Predicting or serializing before fit raises
`RuntimeError`; invalid configuration/schema raises an error.

## Historical data and cutoff contract

Required columns: `date`, `home`, `away`, `home_goals`, `away_goals`. Team identities
are exact, nonempty strings. Normalize aliases and deduplicate fixtures upstream;
duplicate observations otherwise count as repeated evidence. Both scores must be
finite nonnegative integers, including a legitimate 0–0. Missing, negative,
fractional or infinite scores and invalid team names are excluded with
`DataWarning`. Invalid dates are excluded. Inputs are copied, never modified.

All timestamps are interpreted in UTC; naive timestamps mean UTC. Only records
with `date < cutoff` are used; equality is excluded. A date-only cutoff is midnight
at the start of that day, so that day's matches are excluded. Training is sorted
canonically for reproducibility and future rows cannot add teams or change priors.

The five-column interface assumes supplied scores are **final observed results**
and `date` represents their availability, or that the caller has already applied
a conservative completion filter. A kickoff timestamp and final score alone do
not reveal when the result was knowable. For stronger protection supply optional
`completed_at`: then both `date < cutoff` and `date <= completed_at < cutoff` are
required. Missing/invalid `completed_at` excludes that observation. This check is
also applied at the internal evaluation split. In-progress numeric scores must
be masked upstream; arbitrary `status` strings are not interpreted. Retrospective
data corrections cannot be detected without upstream data snapshots.

## Estimation and priors

For an ordinary home fixture:

```text
log(lambda_home) = intercept + home_advantage + attack[home] - defence[away]
log(lambda_away) = intercept                  + attack[away] - defence[home]
weight(match) = 2 ** (-age_in_days_at_training_cutoff / half_life_days)
```

Higher defence means fewer conceded goals. The SciPy L-BFGS-B optimizer minimizes
the weighted joint Poisson negative log likelihood (omitting parameter-independent
factorials) plus `0.5 * regularization * sum(attack² + defence²)`. Intercept and
home advantage have quadratic penalties of strength `prior_matches` around
`log(prior_goals)` and `prior_home_advantage`. Positive penalties resolve the
unpenalized attack/defence identifiability ambiguity and shrink sparse teams toward
zero. Likelihood weights are not renormalized: older evidence contributes less
relative to the fixed priors. Analytic gradients avoid finite-difference cost.

Intercept is bounded to [-3, 3], home advantage to [-1.5, 1.5], and each team
coefficient to [-2.5, 2.5]. Predictions are capped to [1e-6, 100] goals for numeric
safety. Unconverged optimization emits and records a warning and uses the fitted
baseline; `fit_info.success`, `fallback`, iterations and objective expose this.
Empty training data uses priors and records `prior_only`, with an explicit coverage
warning. This allows finite predictions without implying evidence exists.

The baseline estimates separately decayed home/away league scoring rates using
`prior_matches` pseudo-observations at `prior_goals * exp(prior_home_advantage)`
and `prior_goals`. Team effects are zero. At neutral venues, both models remove
home advantage; the baseline therefore uses its away/neutral rate for both sides.
Training assumes ordinary home fixtures; the required schema does not model
neutral historical venues, extra time, red cards or lineups. Use consistent
regulation-time scores upstream.

Unknown teams receive zero attack and defence (population prior); each unknown
prediction emits `CoverageWarning`. Teams with fewer than `min_team_matches`
**decayed matches** also warn, including stale teams with many old observations.
Messages persist uniquely in `.warnings` and serialization; prediction may append
messages but never changes learned strengths. Coverage includes raw match counts
and summed decay weights. These warnings also apply to the baseline to make gaps
visible. No missing club receives invented team-specific strength.

Serialization includes schema version, model type, all constructor parameters,
training cutoff/latest training date, raw and weighted training counts, team
coverage, learned coefficients, optimizer diagnostics and warning messages.
The returned dictionary is a deep copy. Identical sorted data/configuration yield
deterministic estimates in the same numerical environment; cross-version bitwise
identity of a fresh SciPy fit is not guaranteed. Saved coefficients avoid refitting.

## Temporal evaluation

Evaluation first filters observations strictly before the requested cutoff. It
uses the earliest approximately 80% as training and the later 20% as holdout,
adjusted for minimum counts. Equal timestamps remain together on one side. Both
candidates fit once with the holdout's starting timestamp as their training cutoff.
They remain frozen throughout the holdout; no held-out scores update strengths,
decay weights, parameters or calibration. `model_params` controls the Poisson
candidate; the baseline shares its decay and population prior settings. Passing
`model_params.kind='baseline'` is rejected because this function compares both.

At most the first `max_holdout` observations after the split are scored, keeping a
contiguous evaluation horizon and limiting cost. The report exposes truncation;
training itself is not capped. A synthetic 20,000-match/320-club smoke check took
about 1.5 seconds to fit and 5.3 seconds to evaluate 2,000 held-out matches on the
development machine; these are illustrative timings, not performance guarantees.

Report fields:

| Field | Meaning |
| --- | --- |
| `status` | `ok` or `insufficient_data` |
| `selected_model`, `recommended_model` | Equal strings: `poisson` or `baseline` |
| `selection_metric` | `log_loss`, evaluated on home/draw/away outcomes |
| `models.poisson`, `models.baseline` | Comparison metrics and coverage diagnostics |
| `n_available`, `n_train`, `n_holdout` | Eligible records and actual split counts |
| `cutoff`, `train_cutoff`, `train_last_date` | UTC boundaries; latter two on successful evaluation |
| `holdout_start`, `holdout_end`, `holdout_truncated` | Actual evaluation horizon |
| `model_params` | Candidate fit configuration on successful evaluation |
| `reason` | Why an insufficient-data evaluation could not run |

Each candidate reports mean natural-log 1X2 `log_loss` (probabilities floored at
1e-15 for this metric), multiclass `brier` (sum of three squared errors per match,
range 0–2), and joint exact-score Poisson `goal_log_loss` including factorials.
Lower is better for all three. Home/draw/away probabilities use the Skellam
distribution, integrating all score tails instead of dropping a finite grid tail.

`calibration_bins` has one list for each outcome. Each equal-width bin includes
`lower`, `upper`, `count`, `mean_probability`, and `observed_frequency`; empty bins
have null means/frequencies. Bins are left-closed/right-open except the last,
which includes one. These are descriptive bins, not a fitted calibrator. The report
also includes `missing_teams`, `sparse_teams`, and `fit_info` for each candidate.

Poisson is selected only if optimization succeeded and its held-out 1X2 log loss
is strictly lower. Ties and insufficient data select baseline. On insufficient
data there are no fabricated metrics (`models={}`); the baseline flag is a
conservative default, not an empirical win. Selection never changes the default
constructor and never fits a final production model. The caller explicitly fits
its chosen kind on all eligible history afterwards.

A single selection holdout does not establish reliability, calibrated probabilities
or unbiased future performance. Choosing a model on these scores consumes the
holdout for selection; use an additional untouched future interval for final
assessment. Domestic leagues and UCL may differ in scoring environments and
schedule strength, especially with weak cross-league links. This model has no
competition offsets, parameter-uncertainty intervals, correlated scores or
low-score correction. Check real competition-specific performance and coverage
before making confidence claims.

## Verification

Run `python -m pytest tests/test_models.py -q`. Tests cover attack and defence
learning, home advantage and neutral symmetry, strict cutoff and completion-time
exclusion, future and holdout perturbation isolation, timestamp grouping, input
immutability, deterministic fits, JSON round-trips, baseline compatibility,
normalized outcome tails, decay weights, invalid scores, sparse/empty priors,
persistent warnings, optimizer fallback and temporal metrics.
