"""Monte Carlo football league simulator.

The model intentionally keeps its inputs small: each team has one goal-margin
power rating (lower = stronger), and fixtures may be supplied in any order. It
uses a bivariate Poisson score model with an optional draw adjustment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Team:
    name: str
    rating: float = 1.5


@dataclass(frozen=True)
class Fixture:
    home: str
    away: str


@dataclass(frozen=True)
class MatchProbabilities:
    """Probabilities expressed as percentages, not betting prices."""
    home_goals: float
    away_goals: float
    home_win_pct: float
    draw_pct: float
    away_win_pct: float
    total_goals_pct: dict[str, float]
    supremacy_pct: dict[str, float]


@dataclass(frozen=True)
class ScoreMatrix:
    """A finite, normalised scoreline distribution for one fixture."""
    home_goals: float
    away_goals: float
    scores: tuple[tuple[int, int, float], ...]
    cumulative_probabilities: tuple[float, ...]


def poisson_probability(goals: int, mean: float) -> float:
    return math.exp(-mean) * mean**goals / math.factorial(goals)


def read_teams(path: str | Path) -> dict[str, Team]:
    """Read a JSON array of {name, rating} objects."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    teams = {row["name"]: Team(row["name"], float(row.get("rating", 1.5))) for row in data}
    if not teams or len(teams) != len(data):
        raise ValueError("Teams must be a non-empty list with unique names.")
    return teams


def read_fixtures(path: str | Path, teams: dict[str, Team]) -> list[Fixture]:
    """Read a CSV with `home` and `away` column headers."""
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows or not rows[0].get("home") or not rows[0].get("away"):
        raise ValueError("Fixture CSV needs home and away columns and at least one row.")
    fixtures = [Fixture(row["home"], row["away"]) for row in rows]
    unknown = {name for f in fixtures for name in (f.home, f.away) if name not in teams}
    if unknown:
        raise ValueError(f"Fixture references unknown teams: {', '.join(sorted(unknown))}")
    if any(f.home == f.away for f in fixtures):
        raise ValueError("A team cannot play itself.")
    return fixtures


class LeagueSimulator:
    def __init__(self, teams: dict[str, Team], *, base_goals: float = 1.35,
                 home_advantage: float = 0.20, score_correlation: float = 0.0,
                 draw_inflation: float = 1.098) -> None:
        if base_goals <= 0 or not 0 <= score_correlation < 1 or draw_inflation <= 0:
            raise ValueError("Invalid goal, correlation, or draw-adjustment parameter.")
        self.teams = teams
        self.base_goals = base_goals
        self.home_advantage = home_advantage
        self.score_correlation = score_correlation
        self.draw_inflation = draw_inflation
        self._score_matrices: dict[Fixture, ScoreMatrix] = {}

    def _score_matrix(self, fixture: Fixture) -> ScoreMatrix:
        """Build a bivariate Poisson scoreline distribution for a fixture.

        `score_correlation` is the target positive correlation between the two
        goal counts. `draw_inflation` multiplies diagonal scorelines (0-0,
        1-1, and so on) before the matrix is normalised.
        """
        if fixture in self._score_matrices:
            return self._score_matrices[fixture]
        home, away = self.teams[fixture.home], self.teams[fixture.away]
        expected_supremacy = away.rating - home.rating + self.home_advantage
        expected_total = max(self.base_goals * 2, abs(expected_supremacy) + 0.6)
        target_home_goals = (expected_total + expected_supremacy) / 2
        target_away_goals = (expected_total - expected_supremacy) / 2

        # A bivariate Poisson is U + W and V + W, where W is a shared scoring
        # component. The cap keeps each independent component non-negative.
        shared_rate = min(
            self.score_correlation * math.sqrt(target_home_goals * target_away_goals),
            target_home_goals,
            target_away_goals,
        )
        home_independent = target_home_goals - shared_rate
        away_independent = target_away_goals - shared_rate
        raw_scores: list[tuple[int, int, float]] = []
        for home_score in range(13):
            for away_score in range(13):
                probability = sum(
                    poisson_probability(shared, shared_rate)
                    * poisson_probability(home_score - shared, home_independent)
                    * poisson_probability(away_score - shared, away_independent)
                    for shared in range(min(home_score, away_score) + 1)
                )
                raw_scores.append((home_score, away_score, probability))
        raw_scores = [
            (h, a, probability * (self.draw_inflation if h == a else 1.0))
            for h, a, probability in raw_scores
        ]
        normalising_total = sum(probability for _, _, probability in raw_scores)
        scores = tuple((h, a, probability / normalising_total) for h, a, probability in raw_scores)
        cumulative: list[float] = []
        running_total = 0.0
        for _, _, probability in scores:
            running_total += probability
            cumulative.append(running_total)
        matrix = ScoreMatrix(
            sum(h * probability for h, _, probability in scores),
            sum(a * probability for _, a, probability in scores),
            scores,
            tuple(cumulative),
        )
        self._score_matrices[fixture] = matrix
        return matrix

    def match_probabilities(self, fixture: Fixture) -> MatchProbabilities:
        """Return expected goals and outcome probabilities for one fixture."""
        matrix = self._score_matrix(fixture)
        home_win = draw = away_win = 0.0
        total_goals = defaultdict(float)
        supremacy = defaultdict(float)
        for hg, ag, probability in matrix.scores:
            total_goals[str(min(hg + ag, 7)) + ("+" if hg + ag >= 7 else "")] += probability
            goal_difference = hg - ag
            difference_label = str(max(-3, min(goal_difference, 3)))
            if goal_difference <= -3:
                difference_label = "-3 or less"
            elif goal_difference >= 3:
                difference_label = "+3 or more"
            elif goal_difference > 0:
                difference_label = f"+{goal_difference}"
            supremacy[difference_label] += probability
            if hg > ag:
                home_win += probability
            elif hg == ag:
                draw += probability
            else:
                away_win += probability
        return MatchProbabilities(
            matrix.home_goals, matrix.away_goals,
            as_percentage(home_win), as_percentage(draw), as_percentage(away_win),
            {label: as_percentage(value) for label, value in sorted(total_goals.items(), key=lambda item: int(item[0].rstrip("+")))},
            {label: as_percentage(supremacy[label]) for label in ("-3 or less", "-2", "-1", "0", "+1", "+2", "+3 or more")},
        )

    def simulate_match(self, fixture: Fixture, rng: random.Random) -> tuple[int, int]:
        matrix = self._score_matrix(fixture)
        draw = rng.random()
        for index, cumulative_probability in enumerate(matrix.cumulative_probabilities):
            if draw <= cumulative_probability:
                home_score, away_score, _ = matrix.scores[index]
                return home_score, away_score
        home_score, away_score, _ = matrix.scores[-1]
        return home_score, away_score

    def simulate_seasons(self, fixtures: Iterable[Fixture], simulations: int,
                         *, seed: int | None = None, top_n: int = 4,
                         relegation_n: int = 3) -> dict:
        if simulations < 1:
            raise ValueError("simulations must be at least 1.")
        if top_n < 0 or relegation_n < 0:
            raise ValueError("Qualification and relegation place counts cannot be negative.")
        fixtures = list(fixtures)
        rng = random.Random(seed)
        stats = {name: defaultdict(int) for name in self.teams}
        total_points = defaultdict(int)
        for _ in range(simulations):
            table = {name: {"points": 0, "gf": 0, "ga": 0, "wins": 0} for name in self.teams}
            for fixture in fixtures:
                hg, ag = self.simulate_match(fixture, rng)
                h, a = table[fixture.home], table[fixture.away]
                h["gf"] += hg; h["ga"] += ag; a["gf"] += ag; a["ga"] += hg
                if hg > ag:
                    h["points"] += 3; h["wins"] += 1
                elif ag > hg:
                    a["points"] += 3; a["wins"] += 1
                else:
                    h["points"] += 1; a["points"] += 1
            ranked = sorted(self.teams, key=lambda n: (table[n]["points"], table[n]["gf"] - table[n]["ga"], table[n]["gf"], table[n]["wins"], rng.random()), reverse=True)
            for position, name in enumerate(ranked, 1):
                stats[name]["position_total"] += position
                stats[name]["champion"] += position == 1
                stats[name]["top_n"] += position <= min(top_n, len(ranked))
                stats[name]["relegated"] += position > len(ranked) - min(relegation_n, len(ranked))
                total_points[name] += table[name]["points"]
        return {"simulations": simulations, "teams": [{
            "team": name,
            "title_pct": as_percentage(stats[name]["champion"] / simulations),
            "top_n_pct": as_percentage(stats[name]["top_n"] / simulations),
            "relegation_pct": as_percentage(stats[name]["relegated"] / simulations),
            "average_position": stats[name]["position_total"] / simulations,
            "average_points": total_points[name] / simulations,
        } for name in sorted(self.teams, key=lambda n: stats[n]["champion"], reverse=True)]}


def as_percentage(probability: float) -> float:
    return round(probability * 100, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate a football league from ratings and fixtures.")
    parser.add_argument("teams", help="Path to teams JSON")
    parser.add_argument("fixtures", help="Path to fixtures CSV")
    parser.add_argument("--simulations", "-n", type=int, default=10_000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--top-n", type=int, default=4, help="Number of qualification places (default: 4)")
    parser.add_argument("--relegation-n", type=int, default=3, help="Number of relegation places (default: 3)")
    parser.add_argument("--score-correlation", type=float, default=0.0, help="Bivariate score correlation, from 0 to under 1")
    parser.add_argument("--draw-inflation", type=float, default=1.098, help="Multiplier for draw scorelines (default: calibrated 1.098)")
    parser.add_argument("--match", nargs=2, metavar=("HOME", "AWAY"), help="Print probabilities for one match instead")
    parser.add_argument("--output", help="Write league results as JSON to this path")
    args = parser.parse_args()
    teams = read_teams(args.teams)
    simulator = LeagueSimulator(
        teams, score_correlation=args.score_correlation, draw_inflation=args.draw_inflation,
    )
    if args.match:
        probabilities = simulator.match_probabilities(Fixture(*args.match))
        print(json.dumps(vars(probabilities), indent=2))
        return
    results = simulator.simulate_seasons(
        read_fixtures(args.fixtures, teams), args.simulations, seed=args.seed,
        top_n=args.top_n, relegation_n=args.relegation_n,
    )
    rendered = json.dumps(results, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
