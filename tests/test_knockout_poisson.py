import pathlib
import shutil
import tempfile
import pandas as pd
import world_cup_simulator as wc
from world_cup_simulator import WorldCupSimulator


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
