"""
Tournament utilities.

DO NOT MODIFY THIS FILE.  Shared by all agents.

Key functions
-------------
train_agents(agents, num_episodes)
    Self-play training: all agents in the list play together and learn.

evaluate_vs_random(agent, num_episodes)
    How well does a single agent do against 2 fully-random opponents?

evaluate_vs_self(agents, num_episodes)
    How well do agents do when playing against each other (epsilon=0)?

run_tournament(matchups, train_episodes, eval_episodes)
    Full pipeline: train each matchup, then rank by combined score.
"""

import sys
import io
import itertools
import random
import numpy as np
from .game import WizardGame
from .base_agent import BaseAgent


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

class _RandomAgent(BaseAgent):
    """Minimal random agent used as dummy opponent in evaluate_vs_random."""
    def act(self, obs, valid_actions):
        return random.choice(valid_actions)


def _play_silent(game: WizardGame) -> list:
    """Run play_episode() with all stdout suppressed. Returns score list."""
    sys.stdout = io.StringIO()
    try:
        return game.play_episode()
    finally:
        sys.stdout = sys.__stdout__


def _freeze_epsilon(agents):
    """Set epsilon=0 on all agents that have one. Returns saved values."""
    saved = []
    for a in agents:
        saved.append(getattr(a, 'epsilon', None))
        if hasattr(a, 'epsilon'):
            a.epsilon = 0.0
    return saved


def _restore_epsilon(agents, saved):
    for a, eps in zip(agents, saved):
        if eps is not None:
            a.epsilon = eps


# ------------------------------------------------------------------
# Core functions
# ------------------------------------------------------------------

def train_agents(agents: list, num_episodes: int):
    """
    Self-play training.

    All agents in the list learn simultaneously by competing against
    each other. As one improves, the others are forced to improve too.
    """
    for _ in range(num_episodes):
        game = WizardGame(agents)
        _play_silent(game)


def evaluate_vs_random(agent: BaseAgent, num_episodes: int = 200) -> float:
    """
    Evaluate one trained agent against 2 random opponents (no exploration).

    Returns the agent's average total score over num_episodes games.
    """
    saved  = _freeze_epsilon([agent])
    scores = []

    for _ in range(num_episodes):
        dummies = [_RandomAgent('R1'), _RandomAgent('R2')]
        game    = WizardGame([agent] + dummies)
        result  = _play_silent(game)
        scores.append(result[0])

    _restore_epsilon([agent], saved)
    return sum(scores) / len(scores)


def evaluate_vs_self(agents: list, num_episodes: int = 200) -> list:
    """
    Evaluate trained agents playing against each other (no exploration).

    Returns a list of average scores, one per agent.
    Fairer than vs_random — agents that learned real strategy will
    score well here, not just ones that exploit random mistakes.
    """
    saved      = _freeze_epsilon(agents)
    all_scores = [[] for _ in agents]

    for _ in range(num_episodes):
        game   = WizardGame(agents)
        result = _play_silent(game)
        for i, s in enumerate(result):
            all_scores[i].append(s)

    _restore_epsilon(agents, saved)
    return [sum(s) / len(s) for s in all_scores]


# ------------------------------------------------------------------
# Grid search
# ------------------------------------------------------------------

def grid_search(AgentClass, param_grid: dict,
                train_episodes: int = 500,
                eval_episodes:  int = 150,
                num_agents:     int = 3) -> list:
    """
    Exhaustive hyperparameter search for AgentClass.

    For every combination in param_grid:
      1. Train num_agents agents in self-play.
      2. Evaluate vs random  (agent 0 vs 2 randoms).
      3. Evaluate vs self    (all agents play each other).
      4. Rank by combined = (vs_rnd + vs_self) / 2.

    Returns results sorted by combined score (best first).
    """
    keys   = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))

    print(f"\n{'='*70}")
    print(f"Grid search: {AgentClass.__name__}  |  {len(combos)} combinations")
    print(f"Train: {train_episodes} eps   Eval: {eval_episodes} eps   Agents: {num_agents}")
    print(f"{'='*70}")
    print(f"  {'params':<45}  {'vs_rnd':>7}  {'vs_self':>7}  {'combined':>8}")
    print(f"  {'-'*45}  {'-'*7}  {'-'*7}  {'-'*8}")

    results = []
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        agents = [AgentClass(f'gs_{j}', **params) for j in range(num_agents)]
        train_agents(agents, train_episodes)

        vs_rnd   = evaluate_vs_random(agents[0], eval_episodes)
        vs_self  = sum(evaluate_vs_self(agents, eval_episodes)) / num_agents
        combined = (vs_rnd + vs_self) / 2

        results.append((params, vs_rnd, vs_self, combined))
        param_str = '  '.join(f'{k}={v}' for k, v in params.items())
        print(f"  [{i:>2}/{len(combos)}] {param_str:<40}  "
              f"{vs_rnd:>7.1f}  {vs_self:>7.1f}  {combined:>8.1f}")

    results.sort(key=lambda x: x[3], reverse=True)

    print(f"\n--- Top 3 (by combined score) ---")
    print(f"  {'params':<45}  {'vs_rnd':>7}  {'vs_self':>7}  {'combined':>8}")
    for rank, (params, vs_rnd, vs_self, combined) in enumerate(results[:3], 1):
        param_str = '  '.join(f'{k}={v}' for k, v in params.items())
        print(f"  #{rank}  {param_str:<42}  {vs_rnd:>7.1f}  {vs_self:>7.1f}  {combined:>8.1f}")

    return results


# ------------------------------------------------------------------
# Full tournament
# ------------------------------------------------------------------

def run_tournament(matchups: list,
                   train_episodes: int = 1000,
                   eval_episodes:  int = 200) -> list:
    """
    Train each matchup, then rank everyone by combined score.

    matchups: list of  (label_string, [agent, agent, agent])
    """
    print(f"\n{'='*70}")
    print(f"Tournament  |  {len(matchups)} matchups")
    print(f"Train: {train_episodes} eps   Eval: {eval_episodes} eps")
    print(f"{'='*70}\n")

    standings = []
    for label, agents in matchups:
        print(f"  Training: {label} ...", end='', flush=True)
        train_agents(agents, train_episodes)

        vs_rnd   = evaluate_vs_random(agents[0], eval_episodes)
        vs_self  = sum(evaluate_vs_self(agents, eval_episodes)) / len(agents)
        combined = (vs_rnd + vs_self) / 2

        standings.append((label, vs_rnd, vs_self, combined))
        print(f"\r  {label:<38}  vs_rnd: {vs_rnd:>6.1f}  "
              f"vs_self: {vs_self:>6.1f}  combined: {combined:>6.1f}")

    standings.sort(key=lambda x: x[3], reverse=True)
    print(f"\n--- Final standings (by combined score) ---")
    print(f"  {'matchup':<38}  {'vs_rnd':>7}  {'vs_self':>7}  {'combined':>8}")
    print(f"  {'-'*38}  {'-'*7}  {'-'*7}  {'-'*8}")
    for rank, (label, vs_rnd, vs_self, combined) in enumerate(standings, 1):
        print(f"  #{rank}  {label:<35}  {vs_rnd:>7.1f}  {vs_self:>7.1f}  {combined:>8.1f}")

    return standings
