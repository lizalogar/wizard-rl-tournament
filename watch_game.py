"""
Watch a trained agent play one full game against 2 random opponents.

Usage:
    python watch_game.py                          # PPO vs 2 random, 1000 training eps
    python watch_game.py --agent dqn              # DQN vs 2 random
    python watch_game.py --agent qtable           # QTable vs 2 random
    python watch_game.py --agent heuristic        # Heuristic vs 2 random (no training)
    python watch_game.py --agent random           # all random (baseline)
    python watch_game.py --agent all              # PPO vs QTable vs DQN
    python watch_game.py --episodes 500           # fewer training episodes (faster)
    python watch_game.py --load ppo_trained.pt    # load saved PPO, skip training
"""

import argparse
from common.game import WizardGame
from common.tournament import train_agents
from agents.agent_ppo       import PPOAgent
from agents.agent_dqn       import DQNAgent
from agents.agent_qtable    import QTableAgent
from agents.agent_heuristic import HeuristicAgent
from agents.random_agent    import RandomAgent


AGENT_CLASSES = {
    'ppo':    PPOAgent,
    'dqn':    DQNAgent,
    'qtable': QTableAgent,
}


def build_agents(agent_type: str, episodes: int, load_path: str = None):
    """Return (focus_agent, all_agents_for_game) after optional training."""
    if agent_type == 'random':
        agents = [RandomAgent(f'Random_{i}') for i in range(3)]
        return agents[0], agents

    if agent_type == 'heuristic':
        h = HeuristicAgent('Heuristic')
        return h, [h, RandomAgent('Random_1'), RandomAgent('Random_2')]

    if agent_type == 'all':
        agents = [PPOAgent('PPO'), DQNAgent('DQN'), QTableAgent('QTable')]
        print(f"Training PPO + DQN + QTable together ({episodes} episodes)...")
        train_agents(agents, episodes)
        return agents[0], agents

    cls  = AGENT_CLASSES[agent_type]
    name = agent_type.upper()
    focus = cls(name)

    if load_path and agent_type == 'ppo':
        focus.load(load_path)
        print(f"Loaded model from {load_path}")
    else:
        train_pool = [cls(f'{name}_{i}') for i in range(3)]
        train_pool[0] = focus
        print(f"Training {name} ({episodes} self-play episodes)...")
        train_agents(train_pool, episodes)

    opponents = [RandomAgent('Random_1'), RandomAgent('Random_2')]
    return focus, [focus] + opponents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent',    default='ppo',
                        choices=['ppo', 'dqn', 'qtable', 'heuristic', 'random', 'all'],
                        help='Which agent to watch (default: ppo)')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Self-play training episodes (default: 1000)')
    parser.add_argument('--load',     type=str, default=None,
                        help='Load a saved PPO model instead of training')
    args = parser.parse_args()

    focus, agents = build_agents(args.agent, args.episodes, args.load)
    print("Ready.\n")

    # Freeze exploration for all learnable agents
    for a in agents:
        if hasattr(a, 'epsilon'):
            a.epsilon = 0.0

    game   = WizardGame(agents, verbose=True)
    scores = game.play_episode()

    # Summary
    ranked = sorted(zip([a.name for a in agents], scores), key=lambda x: -x[1])
    print("Rank  Agent          Score")
    print("----  ----------     -----")
    for rank, (name, score) in enumerate(ranked, 1):
        marker = ' <-- focus' if name == focus.name else ''
        print(f"  {rank}   {name:<14} {score:>6}{marker}")


if __name__ == '__main__':
    main()
