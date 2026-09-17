# Dashboard simplification design

## Goal

Make the Champions 2026/27 dashboard easier to read, add a one-command simulation workflow with a configurable Monte Carlo count, and preserve the existing offline snapshot architecture.

## User-facing layout

The page has three primary sections:

1. A probability table for all 36 teams. The probability columns follow the tournament journey from hardest to easiest: champion, final, semifinal, quarterfinal, round of 16, knockout playoff, top 8, positions 9-16, positions 17-24, eliminated. Expected points and expected league-phase position remain at the end.
2. An expected league-phase ranking from 1 to 36, sorted by expected position and showing each team crest.
3. A historical chart of expected position across saved snapshots. Every team is represented, the y-axis is reversed so first place appears at the top, and interactive hover provides team and snapshot details. Crests are omitted from line endpoints because 36 overlapping images make the chart unreadable; the adjacent ranking provides the crest-based identity layer.

Technical metadata remains available in a collapsed expander so it does not dominate the page.

## Crest handling

Crests are stored under `app/assets/crests/` and rendered from local data URIs so the dashboard remains usable offline. A deterministic initial-based badge is used when an official crest asset is unavailable.

## Simulation command

Add `Nueva-simulacion.ps1` at the project root. The user supplies the number of Monte Carlo simulations, for example:

```powershell
.\Nueva-simulacion.ps1 -Simulaciones 200000
```

The script uses the current local datasets, writes a new immutable snapshot, refreshes the `latest` pointer, and prints the run identifier. The existing matchday update command remains the workflow for downloading new results before simulation.

## Verification

Unit tests cover column ordering, expected-ranking ordering, complete evolution-series construction, crest fallback behavior, and command presence. Streamlit smoke tests verify the three main sections. The full project test suite must pass before delivery.
