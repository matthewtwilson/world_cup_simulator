import pathlib
import shutil
import tempfile
import pandas as pd
from world_cup_simulator import WorldCupSimulator


def test_group_ordering_34():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    fixture = repo_root / 'tests' / 'fixtures' / 'schedule_snapshot.csv'
    with tempfile.TemporaryDirectory() as d:
        sched_dst = pathlib.Path(d) / 'schedule_2026.csv'
        shutil.copy2(fixture, sched_dst)

        sim = WorldCupSimulator(str(sched_dst), str(repo_root / 'groups.json'), str(repo_root / 'third_place_matrix.json'), str(repo_root / 'knockout_setup.json'))
        standings = sim.calculate_group_standings()

        expected = {
            'Group A': ['Mexico', 'South Korea', 'Czech Republic', 'South Africa'],
            'Group B': ['Canada', 'Switzerland', 'Bosnia and Herzegovina', 'Qatar'],
            'Group C': ['Brazil', 'Morocco', 'Scotland', 'Haiti'],
            'Group D': ['United States', 'Australia', 'Paraguay', 'Turkey'],
            'Group E': ['Germany', 'Ivory Coast', 'Ecuador', 'Curacao'],
            'Group F': ['Netherlands', 'Sweden', 'Japan', 'Tunisia'],
            'Group G': ['New Zealand', 'Iran', 'Belgium', 'Egypt'],
            'Group H': ['Uruguay', 'Saudi Arabia', 'Cape Verde', 'Spain'], #Spain should be 3rd here, based on Fifa Ranking, which is not in the simulator yet
            'Group I': ['Norway', 'France', 'Senegal', 'Iraq'],
            'Group J': ['Argentina', 'Austria', 'Jordan', 'Algeria'],
            'Group K': ['Colombia', 'Democratic Republic of the Congo', 'Portugal', 'Uzbekistan'],
            'Group L': ['England', 'Ghana', 'Panama', 'Croatia'],
        }

        for group_name, expected_order in expected.items():
            g = standings[standings['Group'] == group_name]
            actual = list(g['Team'].tolist())
            assert actual == expected_order, (
                f'Expected order for {group_name} to be {expected_order}, got {actual}\n'
                f'{g[["Rank","Team","Pts","GD","GF","FP"]].to_string(index=False)}'
            )


# Pytest will discover and run `test_group_g_ordering`.
