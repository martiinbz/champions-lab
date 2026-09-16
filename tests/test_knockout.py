import copy
import numpy as np
import pandas as pd
import pytest
from champions.knockout import validate_state, conditioned_bracket, _tie


def state():
    teams = [f'T{i:02}' for i in range(36)]
    def slots(starts):
        return [teams[i] for i in starts] + [teams[i+1] for i in starts]
    return dict(source='unit test only', as_of='2027-06-06T00:00:00Z', league_order=teams,
                playoff_seeded=slots([14,8,12,10]), playoff_unseeded=slots([16,22,18,20]),
                round16_seeds=slots([0,6,2,4]), ties={})


def leg(h,a):
    return dict(home_goals=h,away_goals=a,completed_at='2027-03-01T23:00:00Z')


def test_fully_observed_tournament_has_one_certain_champion():
    s=state(); teams=s['league_order']
    for stage,n in [('po',8),('r16',8),('qf',4),('sf',2)]:
        for i in range(n):
            s['ties'][f'{stage}-{i}']={'legs':[leg(0,0),leg(1,0)]}
    s['ties']['final-0']={'legs':[leg(0,1)]}
    validate_state(s, teams, pd.DataFrame({'home_goals':[0],'away_goals':[0]}))
    order=np.broadcast_to(np.arange(36),(100,36))
    rounds=conditioned_bracket(order,np.ones((2,36,36,2)),np.random.default_rng(4),s,teams)
    assert np.all(rounds['champion']==1)
    assert np.all(rounds['final']==[0,1])
    assert all(len(np.unique(row))==16 for row in rounds['round16'])


def test_incomplete_first_leg_is_retained_and_known_winner_validated():
    rates=np.ones((2,2,2,2))
    first=np.zeros(3000,dtype=int); second=np.ones(3000,dtype=int)
    winners=_tie(first,second,rates,np.random.default_rng(1),{'legs':[leg(10,0)]})
    assert (winners==0).mean() > .99
    fixed={'legs':[leg(0,0),leg(0,0)],'_winner_index':1}
    assert (_tie(first,second,rates,np.random.default_rng(1),fixed)==1).all()
    fixed['legs']=[leg(5,0),leg(0,0)]
    with pytest.raises(ValueError,match='contradice'):
        _tie(first,second,rates,np.random.default_rng(1),fixed)


def test_future_draw_and_missing_parent_results_fail_closed():
    s=state(); f=pd.DataFrame({'home_goals':[0],'away_goals':[0]})
    with pytest.raises(ValueError,match='posterior'):
        validate_state(s,s['league_order'],f,pd.Timestamp('2027-01-01',tz='UTC'))
    s['ties']['r16-0']={'legs':[leg(0,0)]}
    with pytest.raises(ValueError,match='anteriores'):
        validate_state(s,s['league_order'],f)
    s=state(); s['ties']['po-0']={'legs':[leg(0,0),leg(0,0)]}
    with pytest.raises(ValueError,match='empate'):
        validate_state(s,s['league_order'],f)


def test_invalid_draw_pair_and_missing_league_results_rejected():
    s=state(); s['playoff_seeded'][0],s['playoff_seeded'][1]=s['playoff_seeded'][1],s['playoff_seeded'][0]
    with pytest.raises(ValueError,match='incompatibles'):
        validate_state(s,s['league_order'],pd.DataFrame({'home_goals':[0],'away_goals':[0]}))
    s=state()
    with pytest.raises(ValueError,match='completada'):
        validate_state(s,s['league_order'],pd.DataFrame({'home_goals':[np.nan],'away_goals':[np.nan]}))


def test_simulation_integrates_official_order_and_knockout_winners():
    from test_simulation import schedule, Model
    from champions.simulation import simulate
    s = state()
    for stage,n in [('po',8),('r16',8),('qf',4),('sf',2)]:
        for i in range(n):
            s['ties'][f'{stage}-{i}']={'legs':[leg(0,0),leg(1,0)]}
    s['ties']['final-0']={'legs':[leg(0,1)]}
    probabilities, diagnostics = simulate(schedule(True), Model(), 100, knockout=s)
    indexed = probabilities.set_index('team')
    assert indexed.loc['T01','champion'] == 1
    assert probabilities.champion.sum() == 1
    assert diagnostics['knockout_conditioned']
    assert diagnostics['unresolved_tie_groups'] == 0
    assert indexed.loc['T00','expected_rank'] == 1
