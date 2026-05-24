"""
Curriculum training for the PPO agent.

Three phases, all using mixed opponent pools to prevent catastrophic forgetting:
  Phase 1 — vs random only            (build basics)
  Phase 2 — 70% heuristic / 30% random (learn tactics without forgetting)
  Phase 3 — 60% self-play / 40% heuristic (refine without forgetting tactics)

Usage:
    python train_ppo.py                         # default episode counts
    python train_ppo.py --p1 300 --p2 800 --p3 800
    python train_ppo.py --save my_ppo.pt
    python train_ppo.py --skip-selfplay         # stop after phase 2
"""

import argparse
import random
import sys
import io
from agents.agent_ppo           import PPOAgent
from agents.agent_heuristic     import HeuristicAgent
from agents.agent_heuristic_pool import (AggressiveBidderAgent,
                                         ConservativeBidderAgent,
                                         TrumpHeavyAgent)
from agents.random_agent        import RandomAgent
from common.game                import WizardGame
from common.tournament          import evaluate_vs_random


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _silent_episode(agents):
    """Play one episode with stdout suppressed."""
    game = WizardGame(agents)
    sys.stdout = io.StringIO()
    try:
        game.play_episode()
    finally:
        sys.stdout = sys.__stdout__


def train_mixed(ppo, opponent_pools, episodes):
    """
    Each episode, randomly pick one pool from opponent_pools and play.
    opponent_pools: list of [opp1, opp2] pairs (agents must be reusable).
    """
    for _ in range(episodes):
        pool = random.choice(opponent_pools)
        _silent_episode([ppo] + pool)


def copy_weights(src: PPOAgent, dst: PPOAgent):
    dst.bid_policy.load_state_dict(src.bid_policy.state_dict())
    dst.play_policy.load_state_dict(src.play_policy.state_dict())
    dst.value_net.load_state_dict(src.value_net.state_dict())


def evaluate(ppo, label, n=200):
    score = evaluate_vs_random(ppo, n)
    print(f"  [{label}]  vs random ({n} games): {score:>7.1f}")
    return score


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--p1',            type=int,  default=500)
    parser.add_argument('--p2',            type=int,  default=1000)
    parser.add_argument('--p3',            type=int,  default=1000)
    parser.add_argument('--save',          type=str,  default='ppo_trained.pt')
    parser.add_argument('--skip-selfplay', action='store_true')
    args = parser.parse_args()

    ppo = PPOAgent('PPO')

    # Reusable stateless opponents
    randoms = [RandomAgent('R1'), RandomAgent('R2')]

    # All four heuristic variants as separate 2-agent pools so each episode
    # the PPO faces two opponents of the same style (cleaner signal per game)
    heu_pools = [
        [HeuristicAgent('H1'),          HeuristicAgent('H2')],
        [AggressiveBidderAgent('Agg1'), AggressiveBidderAgent('Agg2')],
        [ConservativeBidderAgent('Con1'), ConservativeBidderAgent('Con2')],
        [TrumpHeavyAgent('Trp1'),       TrumpHeavyAgent('Trp2')],
    ]
    # Default heuristic pair (still used for phase-3 mix and final eval)
    heuristics = heu_pools[0]

    # ------------------------------------------------------------------
    # Phase 1 — vs random
    # ------------------------------------------------------------------
    print(f"\nPhase 1: {args.p1} episodes vs random")
    train_mixed(ppo, [randoms], args.p1)
    evaluate(ppo, 'after phase 1')

    # ------------------------------------------------------------------
    # Phase 2 — 70 % heuristic pool (equal split across 4 styles), 30 % random
    # ------------------------------------------------------------------
    print(f"\nPhase 2: {args.p2} episodes  (70% heuristic pool / 30% random)")
    pools_p2 = heu_pools * 7 + [randoms] * 3   # each style weighted equally
    train_mixed(ppo, pools_p2, args.p2)
    evaluate(ppo, 'after phase 2')

    # ------------------------------------------------------------------
    # Phase 3 — 60 % self-play, 40 % heuristic pool
    # ------------------------------------------------------------------
    if not args.skip_selfplay:
        print(f"\nPhase 3: {args.p3} episodes  (60% self-play / 40% heuristic pool)")
        ppo2, ppo3 = PPOAgent('PPO_2'), PPOAgent('PPO_3')
        copy_weights(ppo, ppo2)
        copy_weights(ppo, ppo3)
        selfplay = [ppo2, ppo3]
        pools_p3 = [selfplay] * 6 + heu_pools  # 6 self-play : 4 heuristic
        train_mixed(ppo, pools_p3, args.p3)
        evaluate(ppo, 'after phase 3')

    # ------------------------------------------------------------------
    # Save & final comparison
    # ------------------------------------------------------------------
    ppo.save(args.save)

    print("\nFinal comparison vs random (200 games each):")
    benchmarks = [
        ('Random',      RandomAgent('R')),
        ('Heuristic',   HeuristicAgent('H')),
        ('Aggressive',  AggressiveBidderAgent('Agg')),
        ('Conservative',ConservativeBidderAgent('Con')),
        ('TrumpHeavy',  TrumpHeavyAgent('Trp')),
        ('PPO',         ppo),
    ]
    for label, agent in benchmarks:
        score = evaluate_vs_random(agent, 200)
        print(f"  {label:<14}: {score:>7.1f}")


if __name__ == '__main__':
    main()
