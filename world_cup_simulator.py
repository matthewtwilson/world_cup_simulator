import pandas as pd
import numpy as np
import json
import random
import argparse
import sys
import time
from collections import Counter, defaultdict

class WorldCupSimulator:
    def __init__(self, schedule_file, groups_file, matrix_file, knockout_file):
        # Load Data
        self.schedule_orig = pd.read_csv(schedule_file)
        self.schedule = self.schedule_orig.copy()
        
        with open(groups_file, 'r') as f:
            data = json.load(f)
            self.groups = data.get('groups', data)
            
        with open(matrix_file, 'r') as f:
            self.matrix = json.load(f)

        with open(knockout_file, 'r') as f:
            self.knockout_setup = json.load(f)
            
        # Calculate Team Stats from finished games in schedule
        self.team_stats = self._calculate_historical_stats()
        
        # Stats containers
        self.team_results = defaultdict(lambda: {
            'pos1': 0, 'pos2': 0, 'pos3': 0, 'pos4': 0,
            'r32': 0, 'r16': 0, 'r8': 0, 'r4': 0, 'r2': 0, 'winner': 0
        })
        self.r32_opponents = defaultdict(Counter)

    def _calculate_historical_stats(self):
        """Calculates GF/GA means for teams based on 'Finished' matches in the CSV."""
        finished = self.schedule_orig[self.schedule_orig['Status'] == 'Finished']
        stats = {}
        
        # Initialize all teams from groups
        all_teams = [team for teams in self.groups.values() for team in teams]
        for team in all_teams:
            stats[team] = {'gf_list': [], 'ga_list': []}
            
        for _, row in finished.iterrows():
            h, a = row['HomeTeam'], row['AwayTeam']
            hg, ag = row['HomeGoals'], row['AwayGoals']
            
            if h in stats:
                stats[h]['gf_list'].append(hg)
                stats[h]['ga_list'].append(ag)
            if a in stats:
                stats[a]['gf_list'].append(ag)
                stats[a]['ga_list'].append(hg)
        
        # Calculate tournament average from finished games
        if not finished.empty:
            total_goals = finished['HomeGoals'].sum() + finished['AwayGoals'].sum()
            self.tournament_avg = total_goals / (len(finished) * 2)
        else:
            self.tournament_avg = 1.3 # Fallback

        final_stats = {}
        for team, data in stats.items():
            gf_mean = np.mean(data['gf_list']) if data['gf_list'] else self.tournament_avg
            ga_mean = np.mean(data['ga_list']) if data['ga_list'] else self.tournament_avg
            final_stats[team] = {'gf_mean': gf_mean, 'ga_mean': ga_mean}
            
        return final_stats

    def _print_pre_simulation_stats(self):
        """Prints statistics about the already finished matches."""
        finished_matches = self.schedule_orig[self.schedule_orig['Status'] == 'Finished']
        
        total_finished = len(finished_matches)
        if total_finished == 0:
            print("No matches have been finished yet.")
            return

        draws = finished_matches[finished_matches['HomeGoals'] == finished_matches['AwayGoals']]
        num_draws = len(draws)
        num_decisions = total_finished - num_draws
        
        total_goals_scored = finished_matches['HomeGoals'].sum() + finished_matches['AwayGoals'].sum()
        avg_goals_per_match = total_goals_scored / total_finished

        print("\n--- Pre-Simulation Stats ---")
        print(f"Total matches finished: {total_finished}")
        print(f"Matches with a draw: {num_draws}")
        print(f"Matches with a decision (win/loss): {num_decisions}")
        print(f"Average goals scored per match: {avg_goals_per_match:.2f}")
        print("----------------------------\n")
        
        # Print current group standings
        standings = self.calculate_group_standings()
        print("--- Current Group Standings ---")
        for group in sorted(standings['Group'].unique()):
            group_standings = standings[standings['Group'] == group]
            print(f"\n{group}:")
            print(f"{'Rank':<6} {'Team':<32} {'Pts':<5} {'GD':<5} {'GF':<5} {'GA':<5} {'FP':<5}")
            print("-" * 60)
            for _, row in group_standings.iterrows():
                print(f"{int(row['Rank']):<6} {row['Team']:<32} {int(row['Pts']):<5} {int(row['GD']):<5} {int(row['GF']):<5} {int(row['GA']):<5} {int(row['FP']):<5}")
        print("-------------------------------\n")


    def reset_schedule(self):
        self.schedule = self.schedule_orig.copy()

    def simulate_match(self, home, away, method='poisson'):
        if method == 'poisson':
            h_gf = self.team_stats.get(home, {}).get('gf_mean', self.tournament_avg)
            h_ga = self.team_stats.get(home, {}).get('ga_mean', self.tournament_avg)
            a_gf = self.team_stats.get(away, {}).get('gf_mean', self.tournament_avg)
            a_ga = self.team_stats.get(away, {}).get('ga_mean', self.tournament_avg)
            
            lambda_home = (h_gf * a_ga) / self.tournament_avg
            lambda_away = (a_gf * h_ga) / self.tournament_avg
            
            h_goals = np.random.poisson(lambda_home)
            a_goals = np.random.poisson(lambda_away)
        else:
            rand = random.random()
            if rand < 0.40: h_goals, a_goals = 1, 0
            elif rand < 0.60: h_goals, a_goals = 1, 1
            else: h_goals, a_goals = 0, 1
        
        return h_goals, a_goals

    def simulate_unfinished_games(self, method='poisson'):
        unfinished_mask = self.schedule['Status'] == 'Unfinished'
        for idx, match in self.schedule[unfinished_mask].iterrows():
            h_goals, a_goals = self.simulate_match(match['HomeTeam'], match['AwayTeam'], method)
            self.schedule.at[idx, 'HomeGoals'] = h_goals
            self.schedule.at[idx, 'AwayGoals'] = a_goals
            self.schedule.at[idx, 'HomeFairPlay'] = -np.random.poisson(0.5)
            self.schedule.at[idx, 'AwayFairPlay'] = -np.random.poisson(0.5)
            self.schedule.at[idx, 'Status'] = 'Finished'

    def calculate_group_standings(self):
        standings = []
        group_matches = self.schedule[self.schedule['Stage'] == 'Group']
        
        for group, teams in self.groups.items():
            for team in teams:
                home_matches = group_matches[group_matches['HomeTeam'] == team]
                away_matches = group_matches[group_matches['AwayTeam'] == team]
                
                gf = home_matches['HomeGoals'].sum() + away_matches['AwayGoals'].sum()
                ga = home_matches['AwayGoals'].sum() + away_matches['HomeGoals'].sum()
                gd = gf - ga
                
                h_wins = len(home_matches[home_matches['HomeGoals'] > home_matches['AwayGoals']])
                a_wins = len(away_matches[away_matches['AwayGoals'] > away_matches['HomeGoals']])
                h_draws = len(home_matches[home_matches['HomeGoals'] == home_matches['AwayGoals']])
                a_draws = len(away_matches[away_matches['AwayGoals'] == away_matches['HomeGoals']])
                
                points = ((h_wins + a_wins) * 3) + ((h_draws + a_draws) * 1)
                fp = home_matches['HomeFairPlay'].sum() + away_matches['AwayFairPlay'].sum()
                
                standings.append({
                    'Group': group, 'Team': team, 'Pts': points, 
                    'GD': gd, 'GF': gf, 'GA': ga, 'FP': fp
                })
                
        df_standings = pd.DataFrame(standings)

        def sort_group_rows(group_df):
            records = group_df.to_dict('records')
            records.sort(key=lambda r: (-r['Pts'], r['Team']))

            sorted_records = []
            start = 0
            while start < len(records):
                end = start + 1
                while end < len(records) and records[end]['Pts'] == records[start]['Pts']:
                    end += 1

                block = records[start:end]
                if len(block) > 1:
                    tied_teams = {r['Team'] for r in block}
                    h2h_matches = group_matches[
                        group_matches['HomeTeam'].isin(tied_teams) &
                        group_matches['AwayTeam'].isin(tied_teams) &
                        (group_matches['Status'] == 'Finished')
                    ]

                    h2h_stats = {team: {'Pts': 0, 'GF': 0, 'GA': 0} for team in tied_teams}
                    for _, match in h2h_matches.iterrows():
                        home, away = match['HomeTeam'], match['AwayTeam']
                        hg, ag = match['HomeGoals'], match['AwayGoals']

                        h2h_stats[home]['GF'] += hg
                        h2h_stats[home]['GA'] += ag
                        h2h_stats[away]['GF'] += ag
                        h2h_stats[away]['GA'] += hg

                        if hg > ag:
                            h2h_stats[home]['Pts'] += 3
                        elif ag > hg:
                            h2h_stats[away]['Pts'] += 3
                        else:
                            h2h_stats[home]['Pts'] += 1
                            h2h_stats[away]['Pts'] += 1

                    for row in block:
                        stats = h2h_stats[row['Team']]
                        row['_h2hPts'] = stats['Pts']
                        row['_h2hGD'] = stats['GF'] - stats['GA']
                        row['_h2hGF'] = stats['GF']

                    block.sort(
                        key=lambda r: (
                            -r['_h2hPts'],
                            -r['_h2hGD'],
                            -r['_h2hGF'],
                            -r['GD'],
                            -r['GF'],
                            -r['FP'],
                            r['Team']
                        )
                    )
                sorted_records.extend(block)
                start = end

            return pd.DataFrame(sorted_records)

        sorted_groups = [sort_group_rows(group_df) for _, group_df in df_standings.groupby('Group', sort=False)]
        df_standings = pd.concat(sorted_groups, ignore_index=True)
        df_standings['Rank'] = df_standings.groupby('Group').cumcount() + 1
        return df_standings

    def get_advancing_teams(self, df_standings):
        top_two = df_standings[df_standings['Rank'] <= 2]
        third_places = df_standings[df_standings['Rank'] == 3].copy()
        third_places = third_places.sort_values(
            by=['Pts', 'GD', 'GF', 'Team'], 
            ascending=[False, False, False, True]
        ).head(8)
        return top_two, third_places

    def get_finished_knockout_result(self, home_team, away_team, stage):
        finished = self.schedule[
            (self.schedule['Stage'] == stage) &
            (self.schedule['Status'] == 'Finished')
        ]

        match = finished[
            (finished['HomeTeam'] == home_team) &
            (finished['AwayTeam'] == away_team)
        ]
        if len(match) == 1:
            row = match.iloc[0]
            return int(row['HomeGoals']), int(row['AwayGoals'])

        match = finished[
            (finished['HomeTeam'] == away_team) &
            (finished['AwayTeam'] == home_team)
        ]
        if len(match) == 1:
            row = match.iloc[0]
            return int(row['AwayGoals']), int(row['HomeGoals'])

        return None

    def run_full_simulation(self, n_simulations=100000, method='fixed'):
        progress_step = max(1, n_simulations // 100)
        start_time = time.perf_counter()
        
        for i in range(n_simulations):
            if i % progress_step == 0:
                elapsed = time.perf_counter() - start_time
                progress = (i / n_simulations) * 100
                sys.stderr.write(f"\rProgress: {progress:.0f}% ({i}/{n_simulations}) | Elapsed: {elapsed:.3f}s")
                sys.stderr.flush()

            self.reset_schedule()
            self.simulate_unfinished_games(method=method)
            standings = self.calculate_group_standings()
            
            # Record group positions
            for _, row in standings.iterrows():
                self.team_results[row['Team']][f'pos{int(row["Rank"])}'] += 1

            top_two, best_thirds = self.get_advancing_teams(standings)
            
            # Slot teams into the bracket
            placeholders = {}
            for _, row in top_two.iterrows():
                placeholders[f"{row['Rank']}{row['Group']}"] = row['Team']
            
            adv_groups_str = "".join(sorted([g.split()[-1] for g in best_thirds['Group'].tolist()]))
            if adv_groups_str in self.matrix:
                mapping = self.matrix[adv_groups_str]
                third_lookup = {row['Group']: row['Team'] for _, row in best_thirds.iterrows()}
                for key, val in mapping.items():
                    target_group = f"Group {val.split()[-1]}"
                    team = third_lookup[target_group]
                    placeholders[f"matrix:{key}"] = team

            # Simulate Knockout Rounds
            match_winners = {}
            for round_info in self.knockout_setup['rounds']:
                round_key = {
                    "Round of 32": "r32", "Round of 16": "r16", "Quarter-finals": "r8",
                    "Semi-finals": "r4", "Final": "r2"
                }.get(round_info['name'])

                for match in round_info['matches']:
                    home_ref = match['home']
                    away_ref = match['away']
                    
                    def resolve(ref):
                        if ref in placeholders: return placeholders[ref]
                        if ref.startswith("Winner "): return match_winners.get(ref.split()[-1])
                        return None
                    
                    home_team = resolve(home_ref)
                    away_team = resolve(away_ref)
                    
                    if home_team and away_team:
                        if round_info['name'] == "Round of 32":
                            self.r32_opponents[home_team][away_team] += 1
                            self.r32_opponents[away_team][home_team] += 1
                            self.team_results[home_team]['r32'] += 1
                            self.team_results[away_team]['r32'] += 1

                        finished_result = self.get_finished_knockout_result(home_team, away_team, round_info['name'])
                        if finished_result is not None:
                            h_goals, a_goals = finished_result
                        else:
                            h_goals, a_goals = self.simulate_match(home_team, away_team, method)

                        if h_goals > a_goals: winner = home_team
                        elif a_goals > h_goals: winner = away_team
                        else: winner = random.choice([home_team, away_team])
                        
                        match_winners[match['id']] = winner
                        
                        if round_info['name'] == "Final":
                            self.team_results[winner]['winner'] += 1
                        
                        next_round_key = {
                            "Round of 32": "r16", "Round of 16": "r8", "Quarter-finals": "r4",
                            "Semi-finals": "r2"
                        }.get(round_info['name'])
                        if next_round_key:
                            self.team_results[winner][next_round_key] += 1
                            
        total_time = time.perf_counter() - start_time
        sys.stderr.write(f"\rProgress: 100% complete. Total time: {total_time:.3f}s            \n")
        sys.stderr.flush()

    def print_stats(self, n_sims):
        print(f"{'Team':<32} | {'1st %':<8} | {'2nd %':<8} | {'3rd %':<8} | {'4th %':<8} | {'R32 %':<8} | {'R16 %':<8} | {'R8 %':<8} | {'R4 %':<8} | {'R2 %':<8} | {'Win %':<8}")
        print("-" * 150)

        def p(val):
            return val / n_sims * 100

        ranked_teams = sorted(
            self.team_results.items(),
            key=lambda item: (
                -p(item[1]['winner']),
                -p(item[1]['r2']),
                -p(item[1]['r4']),
                -p(item[1]['r8']),
                -p(item[1]['r16']),
                -p(item[1]['r32']),
                -p(item[1]['pos1']),
                -p(item[1]['pos2']),
                -p(item[1]['pos3']),
                -p(item[1]['pos4']),
                item[0],
            ),
        )

        for team, stats in ranked_teams:
            print(
                f"{team:<32} | {p(stats['pos1']):>7.2f}% | {p(stats['pos2']):>7.2f}% | {p(stats['pos3']):>7.2f}% | {p(stats['pos4']):>7.2f}% | {p(stats['r32']):>7.2f}% | {p(stats['r16']):>7.2f}% | {p(stats['r8']):>7.2f}% | {p(stats['r4']):>7.2f}% | {p(stats['r2']):>7.2f}% | {p(stats['winner']):>7.2f}%"
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=100000)
    parser.add_argument("--method", type=str, default='fixed', choices=['fixed', 'poisson'])
    args = parser.parse_args()

    sim = WorldCupSimulator('schedule_2026.csv', 'groups.json', 'third_place_matrix.json', 'knockout_setup.json')
    sim._print_pre_simulation_stats() # Call the new method here
    sim.run_full_simulation(n_simulations=args.sims, method=args.method)
    sim.print_stats(args.sims)
    
    print("\nR32 Opponent Frequencies (for United States):")
    for opp, count in sim.r32_opponents['United States'].most_common():
        print(f"  {opp}: {count/args.sims*100:>6.2f}%")
