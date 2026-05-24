"""
Compare two saved PPO checkpoints against common benchmarks.

Usage:
    python compare_models.py --a ppo_trained.pt --b ppo_v2.pt
    python compare_models.py --a ppo_trained.pt --b ppo_v2.pt --games 300
"""

import argparse
import torch
from agents.agent_ppo       import PPOAgent
from agents.agent_heuristic import HeuristicAgent
from common.tournament      import evaluate_vs_random, evaluate_vs_self


def load_ppo(path: str, name: str) -> PPOAgent:
    data = torch.load(path, weights_only=True)
    obs_augment = data.get('obs_augment', False)
    agent = PPOAgent(name, obs_augment=obs_augment)
    agent.load(path)
    agent.epsilon = 0.0
    return agent


def bar(score, lo=-400, hi=300, width=28):
    ratio  = (score - lo) / (hi - lo)
    filled = max(0, min(width, int(round(ratio * width))))
    return '[' + '█' * filled + '░' * (width - filled) + f']  {score:>7.1f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--a',     required=True, help='First checkpoint (e.g. ppo_trained.pt)')
    parser.add_argument('--b',     required=True, help='Second checkpoint (e.g. ppo_v2.pt)')
    parser.add_argument('--games', type=int, default=200, help='Eval games per benchmark')
    args = parser.parse_args()

    n = args.games

    agent_a = load_ppo(args.a, 'Model-A')
    agent_b = load_ppo(args.b, 'Model-B')

    print(f"\n  Model A: {args.a}  (obs_augment={agent_a.obs_augment})")
    print(f"  Model B: {args.b}  (obs_augment={agent_b.obs_augment})")

    # --- vs Random ---
    print(f"\n{'─'*52}")
    print(f"  vs 2 Random opponents  ({n} games each)")
    print(f"{'─'*52}")
    score_a_rnd = evaluate_vs_random(agent_a, n)
    score_b_rnd = evaluate_vs_random(agent_b, n)
    print(f"  {args.a:<22}  {bar(score_a_rnd)}")
    print(f"  {args.b:<22}  {bar(score_b_rnd)}")
    delta_rnd = score_b_rnd - score_a_rnd
    print(f"\n  Δ (B - A): {delta_rnd:+.1f}  {'← B wins' if delta_rnd > 0 else '← A wins' if delta_rnd < 0 else '← tie'}")

    # --- vs 2 Heuristic ---
    print(f"\n{'─'*52}")
    print(f"  vs 2 Heuristic opponents  ({n} games each)")
    print(f"{'─'*52}")
    h1, h2 = HeuristicAgent('H1'), HeuristicAgent('H2')
    scores_a_heu = evaluate_vs_self([agent_a, h1, h2], n)
    h1, h2 = HeuristicAgent('H1'), HeuristicAgent('H2')
    scores_b_heu = evaluate_vs_self([agent_b, h1, h2], n)
    score_a_heu = scores_a_heu[0]
    score_b_heu = scores_b_heu[0]
    print(f"  {args.a:<22}  {bar(score_a_heu)}")
    print(f"  {args.b:<22}  {bar(score_b_heu)}")
    delta_heu = score_b_heu - score_a_heu
    print(f"\n  Δ (B - A): {delta_heu:+.1f}  {'← B wins' if delta_heu > 0 else '← A wins' if delta_heu < 0 else '← tie'}")

    print()


if __name__ == '__main__':
    main()
