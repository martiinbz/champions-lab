# Free model accuracy improvements

## Goal

Improve Champions 2026/27 forecasts using only stable, no-key data sources. The
weekly update must ingest the latest Champions and domestic results, rebuild every
derived rating without future leakage, validate candidate models chronologically,
run Monte Carlo, and preserve a new immutable snapshot.

Current UEFA, Football-Data.co.uk and OpenFootball ingestion remains the source of
truth. The current generation already contains 9,424 matches across domestic and
European competitions. No unofficial xG scraper will be added. The normalized data
contract will reserve nullable xG fields so a reliable provider can be integrated
later without changing model interfaces.

## Data contract

The historical match table keeps its existing identity, score, competition and
source fields and adds optional pre-match or match-level fields when an upstream
Football-Data CSV supplies them:

- shots and shots on target;
- bookmaker 1X2 odds, explicitly tagged by bookmaker and timestamp semantics;
- nullable home and away xG fields, unavailable in the first implementation.

Missing optional values remain null and never become zero. Training features must
only use information knowable before the forecast cutoff. Closing odds and match
statistics may be retained for research and historical evaluation, but cannot be
used as pre-match features unless their availability contract proves they were
known before kickoff.

## Dynamic ratings

### Team Elo

Process matches chronologically. Each team starts from a population rating shrunk
toward its league rating. Updates use the expected 1X2 outcome, actual result,
home advantage, goal-difference multiplier and exponential recency weighting.
Ratings used for a match are always the values immediately before that match.

Hyperparameters such as K-factor, home advantage and decay are selected only from
past validation blocks. The production rating is then rebuilt on all eligible
matches before the cutoff.

### League strength

Estimate league offsets from cross-league European matches. Domestic matches update
clubs within a league; Champions, Europa League, Conference League and qualifiers
connect leagues to a common scale. Sparse leagues are regularized toward the
European population mean. League strength is learned from historical results, not
assigned by reputation.

## Match models

Retain the current regularized independent-Poisson model as a candidate and add a
Dixon-Coles candidate. Dixon-Coles uses time-decayed attack and defence strengths,
home advantage, league offsets, pre-match Elo difference and a low-score
correlation correction.

The production forecast is an ensemble of:

1. current regularized Poisson;
2. enhanced Dixon-Coles score probabilities with Elo and league strength.

Ensemble weights are selected with chronological expanding-window validation on
UCL holdouts. They must be nonnegative and sum to one. If evidence is insufficient,
the system falls back to the best validated single model rather than inventing
weights. Promotion requires at least two valid selection folds and a weighted mean
UCL log-loss improvement of at least 0.005 over the current production model. The
newest assessment block remains untouched: it is reported as an honest final check
and never used to tune weights or reverse the selection decision.

The simulator gains an optional `sample_scores` model interface. For each unplayed
match and simulation it selects a component according to the learned ensemble
weights and samples from that component's coherent score distribution. Existing
models keep working through the current independent-Poisson `predict_goals`
fallback. Evaluation mixes the same component score distributions before deriving
1X2 probabilities, so validation and tournament simulation use one definition.

## Weekly workflow

`Actualizar-jornada.ps1` performs one end-to-end run:

1. download current UEFA and domestic results;
2. validate and snapshot raw inputs;
3. rebuild chronological Elo and league strengths;
4. fit and evaluate candidate models without future leakage;
5. select the best model or ensemble using past UCL blocks;
6. simulate the tournament with the requested Monte Carlo count;
7. save model parameters, feature configuration, metrics, hashes and probabilities
   in a new immutable snapshot.

Repeated simulations over unchanged data may create snapshots, but the dashboard
evolution groups identical data states so Monte Carlo noise is not presented as new
sporting information.

## Failure handling

- A failed optional source does not silently become empty data; provenance records
  the failure and existing mandatory coverage rules decide whether the run stops.
- Missing optional statistics do not exclude otherwise valid matches.
- Disconnected leagues, insufficient cross-league evidence, optimizer failure and
  weak UCL holdouts produce explicit diagnostics and conservative fallbacks.
- No candidate is promoted because it performs better on its training data.

## Verification and acceptance

- Unit tests cover chronological Elo updates, league connectivity and shrinkage,
  Dixon-Coles low-score behavior, nullable optional fields and deterministic replay.
- Leakage tests prove that changing future matches cannot affect earlier ratings,
  fitted parameters or validation predictions.
- Expanding-window tests compare log loss, Brier score, calibration error and goal
  log loss against the existing production model.
- The improved model is enabled by default only when at least two valid selection
  folds exist and its weighted mean UCL log loss improves by at least 0.005 over
  the current model. Otherwise the current model remains production default and
  the new candidate's results are recorded for analysis. The untouched final
  assessment block is diagnostic and cannot change that choice.
- Existing tournament probability conservation, snapshot reproducibility and
  dashboard tests remain green.
