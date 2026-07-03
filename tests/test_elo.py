import json
import pathlib
import random
import shutil
import tempfile
import pytest
from world_cup_simulator import WorldCupSimulator


def test_elo_group_stage_match_outcomes(tmp_path):
    """Test Elo-based group stage simulation with possible draws."""
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    fixture = repo_root / 'tests' / 'fixtures' / 'schedule_34.csv'
    sched_dst = tmp_path / 'schedule_2026.csv'
    shutil.copy2(fixture, sched_dst)
    
    # Create a temporary Elo file with known ratings
    elo_dst = tmp_path / 'elo_ratings.json'
    elo_data = {
        'elo_ratings': {
            'Germany': 1800,
            'Paraguay': 1500,
        }
    }
    with open(elo_dst, 'w') as f:
        json.dump(elo_data, f)

    sim = WorldCupSimulator(
        str(sched_dst),
        str(repo_root / 'groups.json'),
        str(repo_root / 'third_place_matrix.json'),
        str(repo_root / 'knockout_setup.json'),
        str(elo_dst),
    )
    
    random.seed(42)
    runs = 10000
    summary = sim.summarize_match_outcomes(
        'Germany',
        'Paraguay',
        n_runs=runs,
        method='elo',
        knockout=False,
    )

    # Verify all three outcomes are possible in group stage (no draws in penalty logic)
    assert summary['ninety_minutes']['home_win'] > 0
    assert summary['ninety_minutes']['draw'] > 0
    assert summary['ninety_minutes']['away_win'] > 0
    assert sum(summary['ninety_minutes'].values()) == runs
    
    # In group stage, extra_time should be identical to ninety_minutes
    assert summary['extra_time'] == summary['ninety_minutes']
    
    # No penalties in group stage
    assert summary['penalties']['home_win'] == 0
    assert summary['penalties']['away_win'] == 0
    
    # Germany (higher Elo) should win more often
    assert summary['ninety_minutes']['home_win'] > summary['ninety_minutes']['away_win']

    # 300 difference should result in a 84.9% chance of a win.
    # assuming 20% chance of a draw first, decisions would be 67.92% win

    home_win_prob = (1.0 / (1.0 + 10.0 ** ((-300.0) / 400.0))) * .8
    
    assert summary['ninety_minutes']['home_win'] == pytest.approx(home_win_prob*runs, rel=5e-2)
    assert summary['ninety_minutes']['draw'] == pytest.approx(.2*runs, rel=5e-2)
    assert summary['ninety_minutes']['away_win'] == pytest.approx((0.8-home_win_prob)*runs, rel=5e-2)


def test_elo_knockout_match_outcomes(tmp_path):
    """Test Elo-based knockout simulation with extra time and penalties."""
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    fixture = repo_root / 'tests' / 'fixtures' / 'schedule_34.csv'
    sched_dst = tmp_path / 'schedule_2026.csv'
    shutil.copy2(fixture, sched_dst)
    
    # Create a temporary Elo file with known ratings
    elo_dst = tmp_path / 'elo_ratings.json'
    elo_data = {
        'elo_ratings': {
            'Germany': 1800,
            'Paraguay': 1500,
        }
    }
    with open(elo_dst, 'w') as f:
        json.dump(elo_data, f)

    sim = WorldCupSimulator(
        str(sched_dst),
        str(repo_root / 'groups.json'),
        str(repo_root / 'third_place_matrix.json'),
        str(repo_root / 'knockout_setup.json'),
        str(elo_dst),
    )
    
    random.seed(42)
    runs=10000
    summary = sim.summarize_match_outcomes(
        'Germany',
        'Paraguay',
        n_runs=runs,
        method='elo',
        knockout=True,
    )

    # Verify outcomes in 90 minutes
    assert summary['ninety_minutes']['home_win'] > 0
    assert summary['ninety_minutes']['draw'] > 0, "Some matches should go to extra time/penalties"
    assert summary['ninety_minutes']['away_win'] > 0
    assert sum(summary['ninety_minutes'].values()) == runs
    
    # Extra time outcomes should sum to runs (only draws from 90 mins go to extra time)
    assert sum(summary['extra_time'].values()) == runs
    
    # Penalties should only be for matches that were draws after 90 minutes
    assert summary['penalties']['home_win'] > 0
    assert summary['penalties']['away_win'] > 0
    total_penalties = summary['penalties']['home_win'] + summary['penalties']['away_win']
    assert total_penalties == summary['ninety_minutes']['draw'], \
        "Penalty count should equal draws from 90 minutes"
    
    # Germany (higher Elo) should win more often overall
    total_home_wins = summary['extra_time']['home_win']
    total_away_wins = summary['extra_time']['away_win']
    assert total_home_wins > total_away_wins


def test_elo_group_stage_vs_knockout_draws(tmp_path):
    """Test that group stage allows draws while knockout doesn't in final result."""
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    fixture = repo_root / 'tests' / 'fixtures' / 'schedule_34.csv'
    sched_dst = tmp_path / 'schedule_2026.csv'
    shutil.copy2(fixture, sched_dst)
    
    # Create a temporary Elo file
    elo_dst = tmp_path / 'elo_ratings.json'
    elo_data = {
        'elo_ratings': {
            'Germany': 1600,
            'Paraguay': 1600,
        }
    }
    with open(elo_dst, 'w') as f:
        json.dump(elo_data, f)

    sim = WorldCupSimulator(
        str(sched_dst),
        str(repo_root / 'groups.json'),
        str(repo_root / 'third_place_matrix.json'),
        str(repo_root / 'knockout_setup.json'),
        str(elo_dst),
    )
    
    random.seed(42)
    
    # Group stage allows draws to be final result
    group_summary = sim.summarize_match_outcomes(
        'Germany',
        'Paraguay',
        n_runs=500,
        method='elo',
        knockout=False,
    )
    group_final_draws = group_summary['extra_time']['draw']
    assert group_final_draws > 0, "Group stage should have draws as final result"
    
    # Knockout: no draws in final result (all resolved by penalties)
    knockout_summary = sim.summarize_match_outcomes(
        'Germany',
        'Paraguay',
        n_runs=500,
        method='elo',
        knockout=True,
    )
    knockout_final_draws = knockout_summary['extra_time']['draw']
    # In knockout, extra_time can show draws (matches going to penalties), but they're all resolved
    # The point is that extra_time['draw'] represents 90+30 minute ties, and penalties determine winner
    assert knockout_final_draws > 0, "Knockout matches can have 90+30 min draws"
    assert knockout_summary['penalties']['home_win'] + knockout_summary['penalties']['away_win'] == knockout_final_draws, \
        "All extra_time draws should be resolved by penalties"
    
    # Verify no matches end in a true draw for knockout - all have a winner via penalties
    assert knockout_summary['extra_time']['home_win'] + knockout_summary['extra_time']['away_win'] + knockout_summary['extra_time']['draw'] == 500
    assert knockout_summary['penalties']['home_win'] > 0
    assert knockout_summary['penalties']['away_win'] > 0

