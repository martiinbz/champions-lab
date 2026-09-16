"""Competition invariants and regulatory edge cases."""
import importlib
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest


def engine():
    assert (Path(__file__).parents[1] / "src/champions/simulation.py").exists(), "simulation engine missing"
    return importlib.import_module("champions.simulation")


def schedule(played=False):
    # Circle schedule: eight rounds, each club exactly four home/four away.
    teams = list(range(36))
    rows = []
    for day in range(8):
        for i in range(18):
            h, a = teams[i], teams[35-i]
            if (day + i) % 2:
                h, a = a, h
            rows.append([f"T{h:02}", f"T{a:02}", 0 if played else np.nan,
                         0 if played else np.nan, "2026-09-08", day+1, "league_phase"])
        teams = [teams[0], teams[-1], *teams[1:-1]]
    # Orient the 8-regular graph using Euler circuits for exactly 4 home/4 away.
    edges = [(r[0], r[1]) for r in rows]
    adjacency = {t: [] for t in set(sum(([h,a] for h,a in edges), []))}
    for j,(h,a) in enumerate(edges):
        adjacency[h].append((j,a)); adjacency[a].append((j,h))
    used = set()
    for start in adjacency:
        stack = [start]
        while stack:
            t = stack[-1]
            while adjacency[t] and adjacency[t][-1][0] in used:
                adjacency[t].pop()
            if not adjacency[t]:
                stack.pop(); continue
            j,a = adjacency[t].pop(); used.add(j)
            rows[j][0:2] = [t,a]; stack.append(a)
    return pd.DataFrame(rows, columns=["home","away","home_goals","away_goals","date","matchday","stage"])


class Model:
    def __init__(self, rate=1.3):
        self.rate = rate
        self.calls = 0

    def predict_goals(self, home, away, neutral=False):
        self.calls += 1
        return self.rate, self.rate


def test_conservation_reproducibility_and_rate_cache():
    m = engine(); f = schedule(); model = Model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p,d = m.simulate(f, model, 128, 7)
        q,e = m.simulate(f.sample(frac=1, random_state=4), Model(), 128, 7)
    pd.testing.assert_frame_equal(p,q)
    assert d == e
    assert model.calls == 2*36*35
    for col,total in dict(top8=8, positions9_16=8, positions17_24=8, playoff=16,
                          eliminated=12, round16=16, quarterfinal=8, semifinal=4, final=2, champion=1).items():
        assert p[col].sum() == pytest.approx(total)
    assert np.allclose(p.top8+p.playoff+p.eliminated, 1)
    assert np.allclose(p.playoff,p.positions9_16+p.positions17_24)
    assert (p.champion <= p.final).all()
    assert p.expected_rank.sum() == pytest.approx(666)


def test_observed_results_preserved_and_missing_tiebreakers_disclosed():
    m = engine(); f = schedule(True); before = f.copy(deep=True)
    with pytest.warns(RuntimeWarning, match="unresolved"):
        p,d = m.simulate(f, Model(0), 32)
    assert (p.expected_points == 8).all()
    assert d["unresolved_tie_groups"] == 32
    assert d["simulations_with_unresolved_ties"] == 32
    assert d["observed_league_matches"] == 144
    pd.testing.assert_frame_equal(f,before)


def test_opponents_aggregate_and_away_goals_order():
    m = engine()
    # Metrics: points, GD, GF, away GF, wins, away wins, opp points/GD/GF.
    metrics = np.zeros((1,4,9)); metrics[0,:,0] = 10
    metrics[0,0,6] = 5; metrics[0,1,6] = 6
    metrics[0,2,3] = 1; metrics[0,3,7] = 100
    order, groups, sims = m._rank(metrics,np.random.default_rng(0))
    assert order.tolist() == [[2,1,0,3]]
    assert groups == sims == 0


def test_bracket_pairings_and_inherited_priority(monkeypatch):
    m = engine(); calls=[]
    def upset(first, second, rates, rng):
        calls.append((first.copy(),second.copy()))
        return first  # Unseeded wins PO and R16, inherits the R16 slot.
    monkeypatch.setattr(m,"_two_leg",upset)
    monkeypatch.setattr(m,"_final",lambda a,b,rates,rng:a)
    rounds = m._bracket(np.arange(36)[None,:],np.zeros((2,36,36,2)),np.random.default_rng(1))
    po_first,po_second=calls[0]
    for lane,(unseeded,seeded) in enumerate([({16,17},{14,15}),({22,23},{8,9}),({18,19},{12,13}),({20,21},{10,11})]):
        assert set(po_first[0,:,lane]) == unseeded
        assert set(po_second[0,:,lane]) == seeded
    assert np.array_equal(calls[2][1],calls[1][0][:,:,[0,2]])
    assert np.array_equal(calls[2][0],calls[1][0][:,:,[1,3]])
    # Our forced QF upset inherits rank-1/2-path SF return priority.
    assert np.array_equal(calls[3][1],calls[2][0][:,:,0])
    assert rounds["champion"].shape == (1,1)


class ScriptedRng:
    def __init__(self, draws): self.draws=iter(draws); self.rates=[]
    def poisson(self, lam): self.rates.append(np.array(lam)); return np.full(np.shape(lam),next(self.draws))
    def random(self, size): return np.full(size,.25)


def test_extra_time_at_return_venue_no_away_goals():
    m=engine(); rates=np.ones((2,2,2,2))*3
    # a home 1-0; b home 2-1: aggregate 2-2, away goal must not settle it.
    rng=ScriptedRng([1,0,2,1,0,1])
    assert m._two_leg(np.array([0]),np.array([1]),rates,rng).tolist()==[0]
    assert len(rng.rates)==6
    assert np.allclose(rng.rates[-1],1)


@pytest.mark.parametrize("criterion", range(9))
def test_every_ranking_criterion_is_lexicographic(criterion):
    m=engine(); metrics=np.zeros((1,2,9))
    metrics[0,0,criterion]=1
    metrics[0,1,criterion+1:]=100
    assert m._rank(metrics,np.random.default_rng(2))[0].tolist()==[[0,1]]


def test_opponents_totals_include_all_opponent_matches():
    m=engine()
    # A-B 2-0, B-C 1-1, C-D 3-0; A's sole opponent B has one point,
    # GD -2 and GF 1. B's opponents A+C have seven points, GD 5 and GF 6.
    h=np.eye(4,dtype=int)[[0,1,2]]; a=np.eye(4,dtype=int)[[1,2,3]]
    metrics=m._league_metrics(np.array([[2,1,3]]),np.array([[0,1,0]]),h,a,h.T@a+a.T@h)
    assert metrics[0,0,6:].tolist()==[1,-2,1]
    assert metrics[0,1,6:].tolist()==[7,5,6]


def test_known_non_draw_scores_determine_expected_points():
    m=engine(); f=schedule(True)
    f.loc[0,["home_goals","away_goals"]]=[4,1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p,_=m.simulate(f,Model(4),5)
    by_team=p.set_index("team")
    assert by_team.loc[f.loc[0,"home"],"expected_points"]==10
    assert by_team.loc[f.loc[0,"away"],"expected_points"]==7
    assert p.expected_points.sum()==289


def test_final_extra_time_and_penalties_use_neutral_rates():
    m=engine(); rates=np.ones((2,2,2,2))*9; rates[1]=3
    rng=ScriptedRng([0,0,1,1])
    assert m._final(np.array([0]),np.array([1]),rates,rng).tolist()==[0]
    assert [v.item() for v in rng.rates]==[3,3,1,1]


def test_counts_across_batch_boundary_and_serializable_diagnostics():
    import json
    m=engine()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p,d=m.simulate(schedule(True),Model(0),1025)
    assert d["unresolved_tie_groups"]==1025
    assert p.champion.sum()==pytest.approx(1)
    assert p.final.sum()==pytest.approx(2)
    json.dumps(d,allow_nan=False)
    assert "undrawn" in " ".join(d["warnings"])


@pytest.mark.parametrize("rate",[-1,np.nan,np.inf])
def test_invalid_model_rates_rejected(rate):
    with pytest.raises(ValueError,match="goal rates"):
        engine().simulate(schedule(),Model(rate),1)


@pytest.mark.parametrize("simulations",[0,-1,1.5,True])
def test_invalid_simulation_count(simulations):
    with pytest.raises(ValueError,match="positive integer"):
        engine().simulate(schedule(),Model(),simulations)


@pytest.mark.parametrize("change",["knockout","half_score","incomplete","negative","duplicate","matchday"])
def test_reject_invalid_or_unsupported_inputs(change):
    m=engine(); f=schedule()
    if change=="knockout": f.loc[0,"stage"]="round16"
    if change=="half_score": f.loc[0,"home_goals"]=1
    if change=="incomplete": f=f.iloc[:-1]
    if change=="negative": f.loc[0,["home_goals","away_goals"]]=[-1,0]
    if change=="duplicate": f.loc[1]=f.loc[0]
    if change=="matchday": f.loc[0,"matchday"]=9
    with pytest.raises((ValueError,NotImplementedError)):
        m.simulate(f,Model(),2)
