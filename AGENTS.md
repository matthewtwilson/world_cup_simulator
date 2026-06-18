# World Cup 2026 Simulator

A data-driven simulation tool for the FIFA World Cup 2026, supporting the expanded 48-team format.

## Overview

This project simulates the entire 2026 World Cup tournament, from the group stages through to the final. It allows for large-scale Monte Carlo simulations to calculate the probabilities of various outcomes for each participating team.

## Key Features

- **2026 Format Support**: Correct implementation of 12 groups of 4 teams, with 32 teams advancing to the knockout rounds (top 2 from each group + 8 best 3rd-place teams).
- **Simulation Methods**:
  - `poisson`: Uses historical goals scored and allowed to simulate realistic match scores using a Poisson distribution.
  - `fixed`: Uses fixed win/draw/loss probabilities for quick estimations.
- **Dynamic Tiebreakers**: Calculates group standings based on points, goal difference, goals scored, and fair play points.
- **Configurable Bracket**: Uses a dedicated `knockout_setup.json` to define the tournament path and `third_place_matrix.json` for complex 3rd-place slotting.
- **Detailed Analytics**:
  - Probability of finishing in each group position (1st-4th).
  - Probability of reaching each knockout stage (R32, R16, QF, SF, Final, Winner).
  - Round of 32 opponent frequency analysis.
- **Performance Tracking**: Millisecond-level timing and progress updates for high-volume simulations.

## Project Structure

- `world_cup_simulator.py`: The core simulation engine.
- `schedule_2026.csv`: The tournament schedule, including pre-filled results for finished games.
- `groups.json`: Team assignments for all 12 groups.
- `knockout_setup.json`: Definition of the knockout bracket structure.
- `third_place_matrix.json`: Mapping for slotting the best 3rd-place teams into the Round of 32.

## Usage

Run a simulation with the desired number of iterations and method:

```bash
python world_cup_simulator.py --sims 100000 --method poisson
```

### Options

- `--sims`: Number of tournament simulations to run (default: 100,000).
- `--method`: Simulation model to use (`poisson` or `fixed`).
