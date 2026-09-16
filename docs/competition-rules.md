# UEFA Champions League 2026/27 simulation contract

The engine implements the men's Champions League league phase and an **undrawn**
knockout bracket. It does not provide a fully conditioned forecast after knockout
draws or matches. Regulations checked against UEFA's 2026/27 edition, effective
29 July 2026; research date: 16 September 2026.

## Interface and supported inputs

```python
probabilities, diagnostics = simulate(fixtures, model, simulations=10000, seed=42)
# Caller, not simulate(), decides whether/where to persist:
probabilities.to_csv("probabilities.csv", index=False)
```

`fixtures` must contain `home`, `away`, `home_goals`, `away_goals`, `date`,
`matchday`, `stage`. Supply the entire 144-match schedule for 36 clubs, eight
matchdays of 18 games, four home/four away per club, eight distinct opponents.
Stage accepts `league` or `league_phase` (spaces/hyphens normalized).
Team strings must be consistent, nonempty and without surrounding whitespace.
Dates must parse; matchdays must be integers 1–8. Both goals must be missing for
an unplayed match, or both finite nonnegative integers for a completed match.
Known scores are fixed in every simulation. The input DataFrame is not mutated.
The caller must mask scores not yet known at its forecast cutoff; the engine has
no cutoff argument and never infers whether an observed score was knowable.

`model.predict_goals(home, away, neutral=False)` returns two nonnegative finite
goal means. The model should be deterministic for reproducibility. All 1,260
ordered distinct pairs are predicted once at home/away and once at neutral venues
(2,520 calls), independent of simulation count. Batches of at most 1,024
simulations bound memory. The same inputs, deterministic model, seed, engine and
NumPy version reproduce the result; fixture row order does not matter. Changing
simulation count or batching need not preserve the sample prefix.

The CSV-ready DataFrame has exactly this column order:

```text
team,top8,positions9_16,positions17_24,playoff,eliminated,round16,quarterfinal,semifinal,final,champion,expected_points,expected_rank
```

Probabilities are fractions, not percentages. `playoff` means reaching the
knockout play-offs (positions 9–24), not winning them. `eliminated` means league
positions 25–36, not eventual failure to win the trophy. `round16` includes top
eight clubs and play-off winners. Expected rank is one-based. Rows are sorted by
team name for stable identity, not by probability.

## League ranking

Wins earn three points, draws one, losses zero. Final positions 1–8 qualify for
R16; 9–24 enter play-offs; 25–36 leave the competition.
[UEFA Article 17](https://documents.uefa.com/r/Regulations-of-the-UEFA-Champions-League-2026/27/Article-17-Match-system-league-phase-Online).

Final standings use descending lexicographic keys: points, goal difference,
goals for, away goals for, wins, away wins, opponents' collective points,
opponents' collective goal difference, opponents' collective goals for. Opponent
totals cover their entire league phase, including their game against the club
being ranked. There is no head-to-head mini-table. The remaining regulatory
keys are disciplinary points (lower first), then club coefficient (higher first).
Cards include players and officials: yellow 1, red 3, two-yellow dismissal 3.
[UEFA Article 18](https://documents.uefa.com/r/Regulations-of-the-UEFA-Champions-League-2026/27/Article-18-Equality-of-points-league-phase-Online).

**Unavailable criteria:** this interface supplies neither disciplinary totals
nor club coefficients. It does not invent them, use alphabetical team order as
an exact resolution, or apply coefficient ahead of unknown discipline. Every
group tied on all available keys is uniformly randomly ordered with the seeded
generator. Diagnostics include `unresolved_tie_groups` (one count per tied group
per simulation), `simulations_with_unresolved_ties`, their simulation fraction,
`missing_tiebreakers`, `tie_fallback` and human-readable `warnings`. A runtime
warning is emitted if any unresolved group occurs. Counts include groups away
from qualification boundaries. This is an approximation, not UEFA's resolution;
even a completed league can therefore have uncertain ranks in the output.

For a future metadata extension, coefficient means the preseason five-season
coefficient over 2021/22–2025/26. Equal coefficients use annual coefficients from
newest backwards, association coefficient, then latest domestic league position.
Current disciplinary totals alone would not settle future disciplinary totals.
[UEFA D.2](https://documents.uefa.com/r/Regulations-of-the-UEFA-Champions-League-2026/27/D.2-Reference-periods-for-rankings-Online),
[UEFA D.8](https://documents.uefa.com/r/Regulations-of-the-UEFA-Champions-League-2026/27/D.8-Equal-coefficients-Online).

## Bracket and return-leg priority

| Play-off seeded pair | Unseeded pair | R16 seeded opponents |
|---|---|---|
| 9/10 | 23/24 | 7/8 |
| 11/12 | 21/22 | 5/6 |
| 13/14 | 19/20 | 3/4 |
| 15/16 | 17/18 | 1/2 |

Each pair is randomly split between silver and blue sides. R16 seeds are then
split between their two designated positions. Within each side, QFs connect the
1/2 and 7/8 paths, and the 3/4 and 5/6 paths. The two QF winners meet in the SF.
The silver and blue winners meet in the final. There is no fresh unrestricted
QF/SF draw. Slot identities persist through upsets.
[UEFA Annex B](https://documents.uefa.com/r/Regulations-of-the-UEFA-Champions-League-2026/27/Annex-B-UEFA-Champions-League-Competition-System-Online).

Play-off seeds and R16 seeds host the return leg in principle. QF return-leg
priority belongs to the paths seeded 1–4; SF priority to paths seeded 1–2.
Eliminating a seed transfers its bracket position and future priority to the
winner, including successive upsets. Surviving original league ranks are never
compared to recalculate this priority. Silver is nominal home in the neutral
final. The simulation assumes standard draw conditions without exceptional UEFA
constraints or venue interventions.
[UEFA Article 19](https://documents.uefa.com/r/Regulations-of-the-UEFA-Champions-League-2026/27/Article-19-Draw-system-knockout-phase-Online).

## Extra time and modelling approximations

After two legs, compare aggregate goals. Equal aggregate triggers two full
15-minute periods at the second-leg venue. Still equal means penalties. Away
goals never break a knockout tie. A drawn first leg alone triggers no extra time.
The neutral final follows the same extra-time/penalty sequence after 90 minutes.
[UEFA Article 21](https://documents.uefa.com/r/Regulations-of-the-UEFA-Champions-League-2026/27/Article-21-Knockout-system-extra-time-and-penalty-shoot-outs-Online),
[UEFA Article 22](https://documents.uefa.com/r/Regulations-of-the-UEFA-Champions-League-2026/27/Article-22-Match-system-final-Online).

Goals are independent Poisson draws using fixed model rates. Extra-time goals
use one-third of the corresponding 90-minute rates with the actual return-leg
home advantage, or neutral rates in the final. Shoot-outs are 50/50. These are
model assumptions, not UEFA rules; no fatigue, score-state adaptation, injuries,
future rating updates or penalty specialists are modelled. Diagnostic
`assumptions` and `warnings` record these limits. Sampling error remains; a
single probability has approximate Monte Carlo standard error
`sqrt(p*(1-p)/simulations)` before model uncertainty.

## Known knockout fixtures and draws: explicitly unsupported

Any supplied non-league row, scored or unplayed, raises `NotImplementedError`
before model prediction. A league-only call always assumes an undrawn bracket;
it cannot detect a real-world draw from omitted information. Diagnostics always
set `knockout_fixture_support=False` and warn against post-draw updates. Do not
strip knockout rows to bypass this guard. There is no separate bracket input in
this version. Future support needs persistent official bracket slots, leg order,
observed scores including extra time/penalties, and conditional draw logic.

## Verification

`python -m pytest tests/test_simulation.py` covers deterministic replay, shuffled
input order, preserved observed scores, all available ranking-key priorities,
opponent aggregates, unresolved-tie counters, regulatory play-off pairings,
QF/SF inherited priority, extra time, neutral final, input validation, prediction
caching, JSON-safe diagnostics and a batch boundary. Probability conservation:
top8=8, positions9_16=8, positions17_24=8, playoff=16, eliminated=12, round16=16,
quarterfinal=8, semifinal=4, final=2, champion=1. Expected ranks sum to 666.
