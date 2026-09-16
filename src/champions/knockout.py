"""Condition forecasts on an official, explicitly sourced knockout bracket.

Slots follow the regulatory tree: side 0 lanes 1/2,7/8,3/4,5/6, then side 1.
Only drawn participants and completed legs belong in the supplied state.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def validate_state(state: dict, teams: list[str], fixtures: pd.DataFrame, cutoff=None) -> None:
    if not isinstance(state, dict) or not state.get('source') or not state.get('as_of'):
        raise ValueError('El cuadro requiere source y as_of verificables.')
    as_of = pd.Timestamp(state['as_of'])
    as_of = as_of.tz_localize('UTC') if as_of.tzinfo is None else as_of.tz_convert('UTC')
    if cutoff is not None and as_of > pd.Timestamp(cutoff):
        raise ValueError('El cuadro contiene información posterior al corte.')
    order = state.get('league_order', [])
    if len(order) != 36 or set(order) != set(teams):
        raise ValueError('league_order debe contener los 36 equipos en su orden oficial final.')
    if fixtures[['home_goals', 'away_goals']].isna().any().any():
        raise ValueError('Un cuadro oficial requiere la fase liga completada.')
    for key, starts in [('playoff_seeded', [14, 8, 12, 10]), ('playoff_unseeded', [16, 22, 18, 20]),
                        ('round16_seeds', [0, 6, 2, 4])]:
        values = state.get(key)
        if values is None and key == 'round16_seeds':
            continue
        if not isinstance(values, list) or len(values) != 8:
            raise ValueError(f'{key} requiere ocho posiciones oficiales.')
        for lane, start in enumerate(starts):
            if {values[lane], values[lane+4]} != set(order[start:start+2]):
                raise ValueError(f'{key}: posiciones incompatibles con el cuadro UEFA.')
    ties = state.get('ties', {})
    allowed = {f'{stage}-{i}' for stage, n in [('po',8), ('r16',8), ('qf',4), ('sf',2), ('final',1)] for i in range(n)}
    if not isinstance(ties, dict) or set(ties) - allowed:
        raise ValueError('Identificador de eliminatoria desconocido.')
    for key, tie in ties.items():
        stage, idx = key.split('-'); idx = int(idx)
        legs = tie.get('legs', [])
        limit = 1 if stage == 'final' else 2
        if not isinstance(legs, list) or len(legs) > limit:
            raise ValueError('Número de partidos de eliminatoria inválido.')
        for leg in legs:
            for score in ('home_goals','away_goals'):
                value = leg.get(score)
                if isinstance(value, bool) or not isinstance(value, (int,float)) or not np.isfinite(value) or value < 0 or int(value) != value:
                    raise ValueError('Cada partido conocido requiere goles enteros no negativos.')
            completed = pd.Timestamp(leg.get('completed_at'))
            if pd.isna(completed) or completed.tzinfo is None or completed > as_of:
                raise ValueError('completed_at debe ser UTC, válido y anterior a as_of.')
        if tie.get('winner') is not None and (len(legs) != limit or tie['winner'] not in teams):
            raise ValueError('Un ganador oficial requiere todos los partidos completados.')
        if len(legs) == limit:
            difference = legs[0]['home_goals'] - legs[0]['away_goals']
            if limit == 2:
                difference -= legs[1]['home_goals'] - legs[1]['away_goals']
            if difference == 0 and not tie.get('winner'):
                raise ValueError('El empate ya completado requiere el ganador oficial tras prórroga/penaltis.')
        if legs and stage != 'po':
            if state.get('round16_seeds') is None:
                raise ValueError('Falta el sorteo de octavos para incorporar resultados posteriores.')
            if stage == 'r16':
                parents = [f'po-{idx}']
            elif stage == 'qf':
                parents = [f'r16-{(idx//2)*4+(idx%2)*2+j}' for j in (0,1)]
            elif stage == 'sf':
                parents = [f'qf-{idx*2+j}' for j in (0,1)]
            else:
                parents = ['sf-0','sf-1']
            if any(len(ties.get(p,{}).get('legs',[])) != 2 for p in parents):
                raise ValueError('Faltan los partidos completados de las rondas anteriores.')


def _tie(first, second, rates, rng, observed, final=False):
    """first hosts the first leg; second hosts return, unless neutral final."""
    legs = observed.get('legs', [])
    lam1 = rates[int(final), first, second]
    if legs:
        a = np.full(first.shape, legs[0]['home_goals'], dtype=np.int64)
        b = np.full(first.shape, legs[0]['away_goals'], dtype=np.int64)
    else:
        a, b = rng.poisson(lam1[...,0]), rng.poisson(lam1[...,1])
    if final:
        et_a, et_b = lam1[...,0], lam1[...,1]
    else:
        lam2 = rates[0, second, first]
        if len(legs) == 2:
            a += legs[1]['away_goals']; b += legs[1]['home_goals']
        else:
            a += rng.poisson(lam2[...,1]); b += rng.poisson(lam2[...,0])
        et_a, et_b = lam2[...,1], lam2[...,0]
    completed = len(legs) == (1 if final else 2)
    known_winner = observed.get('_winner_index')
    if completed:
        if known_winner is not None:
            if not (((first == known_winner) | (second == known_winner)).all()):
                raise ValueError('El ganador conocido no participa en esa eliminatoria.')
            if ((a > b) & (first != known_winner)).any() or ((b > a) & (second != known_winner)).any():
                raise ValueError('El ganador declarado contradice el marcador agregado de 90 minutos.')
            return np.full(first.shape, known_winner)
        if (a == b).any():
            raise ValueError('Falta el ganador oficial del empate completado.')
    else:
        tied = a == b
        a[tied] += rng.poisson(et_a[tied] / 3)
        b[tied] += rng.poisson(et_b[tied] / 3)
    return np.where((a > b) | ((a == b) & (rng.random(a.shape) < .5)), first, second)


def conditioned_bracket(order, rates, rng, state, teams):
    index = {name: i for i,name in enumerate(teams)}
    size = len(order)
    seeded = np.broadcast_to([index[t] for t in state['playoff_seeded']], (size,8))
    unseeded = np.broadcast_to([index[t] for t in state['playoff_unseeded']], (size,8))
    if state.get('round16_seeds') is not None:
        direct = np.broadcast_to([index[t] for t in state['round16_seeds']], (size,8))
    else:
        starts = np.array([0,6,2,4])
        swap = rng.integers(0,2,size=(size,4))
        positions = np.concatenate([starts+swap, starts+1-swap], axis=1)
        direct = np.take_along_axis(order, positions, axis=1)
    def play(stage, first, second, final=False):
        result = []
        for i in range(first.shape[1]):
            known = dict(state.get('ties',{}).get(f'{stage}-{i}', {}))
            if known.get('winner'):
                known['_winner_index'] = index[known['winner']]
            result.append(_tie(first[:,i], second[:,i], rates, rng, known, final))
        return np.stack(result, axis=1)
    po = play('po', unseeded, seeded)
    qf = play('r16', po, direct)
    sf = play('qf', qf[:,[1,3,5,7]], qf[:,[0,2,4,6]])
    finalists = play('sf', sf[:,[1,3]], sf[:,[0,2]])
    champions = play('final', finalists[:,[0]], finalists[:,[1]], True)
    return {'round16':np.concatenate([po,direct],axis=1),'quarterfinal':qf,
            'semifinal':sf,'final':finalists,'champion':champions}
