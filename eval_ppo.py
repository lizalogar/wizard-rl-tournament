"""
Evaluate a saved PPO model against QTable, DQN, Heuristic, and Random.

Usage:
    python eval_ppo.py                          # loads ppo_trained.pt
    python eval_ppo.py --load my_ppo.pt
    python eval_ppo.py --train-opponents 500    # how long to train QTable/DQN (default 1000)
    python eval_ppo.py --watch                  # play one visible game at the end
"""

import argparse
from agents.agent_ppo       import PPOAgent
from agents.agent_dqn       import DQNAgent
from agents.agent_qtable    import QTableAgent
from agents.agent_heuristic import HeuristicAgent
from agents.random_agent    import RandomAgent
from common.tournament      import train_agents, evaluate_vs_random, evaluate_vs_self
from common.game            import WizardGame


def bar(score, lo=-600, hi=200, width=30):
    """Simple ASCII bar for quick visual comparison."""
    ratio = (score - lo) / (hi - lo)
    filled = int(round(ratio * width))
    filled = max(0, min(width, filled))
    return '[' + '█' * filled + '░' * (width - filled) + f']  {score:>7.1f}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load',             default='ppo_trained.pt')
    parser.add_argument('--train-opponents',  type=int, default=1000,
                        help='Self-play episodes to train QTable and DQN (default 1000)')
    parser.add_argument('--eval-games',       type=int, default=200,
                        help='Evaluation games per matchup (default 200)')
    parser.add_argument('--watch',            action='store_true',
                        help='Play one visible game PPO vs QTable vs DQN at the end')
    args = parser.parse_args()

    n = args.eval_games

    # ------------------------------------------------------------------
    # Load PPO
    # ------------------------------------------------------------------
    ppo = PPOAgent('PPO')
    ppo.load(args.load)
    ppo.epsilon = 0.0

    # ------------------------------------------------------------------
    # Train opponents from scratch
    # ------------------------------------------------------------------
    print(f"\nTraining QTable ({args.train_opponents} self-play eps)...", end='', flush=True)
    qt_agents = [QTableAgent(f'QT_{i}') for i in range(3)]
    train_agents(qt_agents, args.train_opponents)
    qt = qt_agents[0];  qt.epsilon = 0.0
    print(" done")

    print(f"Training DQN   ({args.train_opponents} self-play eps)...", end='', flush=True)
    dqn_agents = [DQNAgent(f'DQN_{i}') for i in range(3)]
    train_agents(dqn_agents, args.train_opponents)
    dqn = dqn_agents[0];  dqn.epsilon = 0.0
    print(" done")

    heu = HeuristicAgent('Heuristic')
    rnd = RandomAgent('Random')

    # ------------------------------------------------------------------
    # vs Random  (individual skill ceiling)
    # ------------------------------------------------------------------
    print(f"\n{'─'*55}")
    print(f"  vs 2 Random opponents  ({n} games each)")
    print(f"{'─'*55}")
    results_rnd = {
        'PPO':       evaluate_vs_random(ppo, n),
        'QTable':    evaluate_vs_random(qt,  n),
        'DQN':       evaluate_vs_random(dqn, n),
        'Heuristic': evaluate_vs_random(heu, n),
        'Random':    evaluate_vs_random(rnd, n),
    }
    for name, score in sorted(results_rnd.items(), key=lambda x: -x[1]):
        print(f"  {name:<12}  {bar(score)}")

    # ------------------------------------------------------------------
    # 3-way head-to-head: PPO vs QTable vs DQN
    # ------------------------------------------------------------------
    print(f"\n{'─'*55}")
    print(f"  3-way head-to-head: PPO vs QTable vs DQN  ({n} games)")
    print(f"{'─'*55}")
    agents_3way = [ppo, qt, dqn]
    scores_3way = evaluate_vs_self(agents_3way, n)
    for agent, score in sorted(zip(agents_3way, scores_3way), key=lambda x: -x[1]):
        print(f"  {agent.name:<12}  {bar(score)}")

    # ------------------------------------------------------------------
    # PPO vs 2 Heuristic
    # ------------------------------------------------------------------
    print(f"\n{'─'*55}")
    print(f"  PPO vs 2 Heuristic  ({n} games)")
    print(f"{'─'*55}")
    h1, h2 = HeuristicAgent('H1'), HeuristicAgent('H2')
    scores_vh = evaluate_vs_self([ppo, h1, h2], n)
    for agent, score in zip([ppo, h1, h2], scores_vh):
        print(f"  {agent.name:<12}  {bar(score)}")

    # ------------------------------------------------------------------
    # Optional: watch one game
    # ------------------------------------------------------------------
    if args.watch:
        print(f"\n{'='*55}")
        print("  Watching: PPO vs QTable vs DQN")
        print(f"{'='*55}")
        game = WizardGame([ppo, qt, dqn], verbose=True)
        final = game.play_episode()
        print("\nGame result:")
        for agent, score in sorted(zip([ppo, qt, dqn], final), key=lambda x: -x[1]):
            print(f"  {agent.name:<12}  {score:>6}")


if __name__ == '__main__':
    main()
