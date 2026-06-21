import pathlib
import shutil
import tempfile
import pandas as pd
from world_cup_simulator import WorldCupSimulator


def test_group_g_ordering():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    fixture = repo_root / 'tests' / 'fixtures' / 'schedule_snapshot.csv'
    with tempfile.TemporaryDirectory() as d:
        sched_dst = pathlib.Path(d) / 'schedule_2026.csv'
        shutil.copy2(fixture, sched_dst)

        sim = WorldCupSimulator(str(sched_dst), str(repo_root / 'groups.json'), str(repo_root / 'third_place_matrix.json'), str(repo_root / 'knockout_setup.json'))
        standings = sim.calculate_group_standings()
        g = standings[standings['Group'] == 'Group G']

        actual = list(g['Team'].tolist())
        expected = ['New Zealand', 'Iran', 'Belgium', 'Egypt']
        assert actual == expected, f'Expected {expected}, got {actual}\n{g[["Rank","Team","Pts","GD","GF","FP"]].to_string(index=False)}'


# Pytest will discover and run `test_group_g_ordering`.
