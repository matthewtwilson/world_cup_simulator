import math
import pathlib
import random
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest
import world_cup_simulator as wc
from world_cup_simulator import WorldCupSimulator


def _poisson_outcome_probabilities(lambda_home, lambda_away, max_goals=12):
    probs = {'home_win': 0.0, 'draw': 0.0, 'away_win': 0.0}
    for home_goals in range(max_goals):
        home_prob = math.exp(-lambda_home) * (lambda_home ** home_goals) / math.factorial(home_goals)
        for away_goals in range(max_goals):
            away_prob = math.exp(-lambda_away) * (lambda_away ** away_goals) / math.factorial(away_goals)
            joint_prob = home_prob * away_prob
            if home_goals > away_goals:
                probs['home_win'] += joint_prob
            elif home_goals < away_goals:
                probs['away_win'] += joint_prob
            else:
                probs['draw'] += joint_prob
    return probs


def test_knockout_poisson_match_uses_extra_time(monkeypatch):
    # This test exercises the knockout-specific Poisson path.
    # A draw after the first 90 minutes should trigger extra time, and
    # a draw after extra time should resolve via a random penalty-kick winner.
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    fixture = repo_root / 'tests' / 'fixtures' / 'schedule_34.csv'
    with tempfile.TemporaryDirectory() as d:
        sched_dst = pathlib.Path(d) / 'schedule_2026.csv'
        shutil.copy2(fixture, sched_dst)

        sim = WorldCupSimulator(
            str(sched_dst),
            str(repo_root / 'groups.json'),
            str(repo_root / 'third_place_matrix.json'),
            str(repo_root / 'knockout_setup.json'),
        )

        calls = []

        def fake_poisson(lam):
            # The first two calls represent the normal 90-minute simulation.
            # The next two calls represent extra-time, which should use lambdas
            # reduced to one-third of the original values.
            calls.append(lam)
            if len(calls) <= 2:
                return 1
            return 0

        monkeypatch.setattr(wc.np.random, 'poisson', fake_poisson)

        # The first simulation should end in a draw, which forces extra time,
        # and the extra-time result should still be a draw, which then resolves
        # via a penalty-kick winner.
        home_goals, away_goals = sim.simulate_match('Germany', 'Paraguay', method='poisson', knockout=True)

        # The penalty-kick branch should produce exactly one goal for the winner.
        assert home_goals + away_goals == 1
        assert len(calls) == 4
        assert calls[2] == calls[0] / 3
        assert calls[3] == calls[1] / 3


def test_explicit_penalty_winner_is_honored_for_1_1_knockout(tmp_path):
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    fixture = repo_root / 'tests' / 'fixtures' / 'schedule_34.csv'
    sched_dst = tmp_path / 'schedule_2026.csv'
    shutil.copy2(fixture, sched_dst)

    schedule = pd.read_csv(sched_dst)
    schedule['PenaltyWinner'] = None
    schedule = pd.concat(
        [
            schedule,
            pd.DataFrame([
                {
                    'MatchID': 999,
                    'Stage': 'Round of 32',
                    'Group': '',
                    'Date': '2026-06-28',
                    'HomeTeam': 'Germany',
                    'AwayTeam': 'Paraguay',
                    'HomeGoals': 1,
                    'AwayGoals': 1,
                    'HomeFairPlay': 0,
                    'AwayFairPlay': 0,
                    'Status': 'Finished',
                    'PenaltyWinner': 'Germany',
                }
            ])
        ],
        ignore_index=True,
    )
    schedule.to_csv(sched_dst, index=False)

    sim = WorldCupSimulator(
        str(sched_dst),
        str(repo_root / 'groups.json'),
        str(repo_root / 'third_place_matrix.json'),
        str(repo_root / 'knockout_setup.json'),
    )

    winner = sim.resolve_knockout_winner('Germany', 'Paraguay', 'Round of 32')

    assert winner == 'Germany'


def test_simulate_match_outcomes_distribution_for_same_game(tmp_path):
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    fixture = repo_root / 'tests' / 'fixtures' / 'schedule_34.csv'
    sched_dst = tmp_path / 'schedule_2026.csv'
    shutil.copy2(fixture, sched_dst)

    sim = WorldCupSimulator(
        str(sched_dst),
        str(repo_root / 'groups.json'),
        str(repo_root / 'third_place_matrix.json'),
        str(repo_root / 'knockout_setup.json'),
    )
    sim.tournament_avg = 1.5
    sim.team_stats = {
        'Germany': {'gf_mean': 2.0, 'ga_mean': 0.0},
        'Paraguay': {'gf_mean': 2.0 / 3.0, 'ga_mean': 2.0 / 3.0},
    }

    np.random.seed(7)
    random.seed(7)

    summary = sim.summarize_match_outcomes(
        'Germany',
        'Paraguay',
        n_runs=1000,
        method='poisson',
        knockout=True,
    )

    assert summary['home_team']['gf_average'] == pytest.approx(2.0, abs=0.2)
    assert summary['home_team']['ga_average'] == pytest.approx(0.0, abs=0.2)
    assert summary['away_team']['gf_average'] == pytest.approx(2.0 / 3.0, abs=0.2)
    assert summary['away_team']['ga_average'] == pytest.approx(2.0 / 3.0, abs=0.2)

    assert sum(summary['ninety_minutes'].values()) == 1000
    assert sum(summary['extra_time'].values()) == 1000
    assert summary['penalties']['home_win'] + summary['penalties']['away_win'] > 0

    home_lambda = (2.0 * (2.0 / 3.0)) / 1.5
    away_lambda = (2.0 / 3.0 * 0.0) / 1.5
    ninety_prob = _poisson_outcome_probabilities(home_lambda, away_lambda)

    assert summary['ninety_minutes']['home_win'] == pytest.approx(ninety_prob['home_win'] * 1000, abs=60)
    assert summary['ninety_minutes']['draw'] == pytest.approx(ninety_prob['draw'] * 1000, abs=60)
    assert summary['ninety_minutes']['away_win'] == pytest.approx(ninety_prob['away_win'] * 1000, abs=60)

    extra_prob = _poisson_outcome_probabilities(home_lambda / 3, away_lambda / 3)
    expected_extra_time = {
        'home_win': ninety_prob['home_win'] * 1000 + ninety_prob['draw'] * 1000 * extra_prob['home_win'],
        'draw': ninety_prob['draw'] * 1000 * extra_prob['draw'],
        'away_win': ninety_prob['away_win'] * 1000 + ninety_prob['draw'] * 1000 * extra_prob['away_win'],
    }

    assert summary['extra_time']['home_win'] == pytest.approx(expected_extra_time['home_win'], abs=60)
    assert summary['extra_time']['draw'] == pytest.approx(expected_extra_time['draw'], abs=60)
    assert summary['extra_time']['away_win'] == pytest.approx(expected_extra_time['away_win'], abs=60)
    assert summary['penalties']['home_win'] + summary['penalties']['away_win'] == pytest.approx(expected_extra_time['draw'], abs=60)
