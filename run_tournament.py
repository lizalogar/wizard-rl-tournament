"""
run_tournament.py  —  entry point for the shared project.

Run from the project root:
    python run_tournament.py

Each colleague implements their agent in agents/agent_*.py.
Common code in common/ is SHARED — do not modify it.
"""

import sys
import io
from agents.agent_qtable import QTableAgent
from agents.agent_dqn    import DQNAgent, DQNAgentSplit, DQNAgentShared
from agents.agent_ppo    import PPOAgent
from agents.random_agent import RandomAgent
from common.game         import WizardGame
from common.tournament   import (train_agents, evaluate_vs_random,
                                  evaluate_vs_self, run_tournament, grid_search)


# ================================================================
# Optional: grid search to find best hyperparameters per agent
# (comment out if you just want a quick tournament run)
# ================================================================

# print("\n>>> QTableAgent grid search")
# qt_results = grid_search(
#     QTableAgent,
#     param_grid={
#         'alpha':         [0.05, 0.1, 0.2],
#         'gamma':         [0.8, 0.9],
#         'epsilon_decay': [0.995, 0.997, 0.999],
#     },
#     train_episodes=500,
#     eval_episodes=100,
# )
# qt_best = qt_results[0][0]

# print("\n>>> DQNAgent grid search")
# dqn_results = grid_search(
#     DQNAgent,
#     param_grid={
#         'lr':            [0.0005, 0.001],
#         'epsilon_decay': [0.995, 0.997],
#         'hidden_size':   [32, 64],
#     },
#     train_episodes=500,
#     eval_episodes=100,
# )
# dqn_best = dqn_results[0][0]


# ================================================================
# Tournament
# ================================================================

run_tournament(
    matchups=[
        # --- Baselines ---
        ("Random x3",
         [RandomAgent(f'R{i}') for i in range(3)]),

        # --- Each agent type playing against itself (self-play) ---
        ("QTable vs QTable vs QTable",
         [QTableAgent(f'QT_{i}') for i in range(3)]),

        ("DQN baseline vs itself",
         [DQNAgent(f'DQN_{i}') for i in range(3)]),

        ("DQN split-input vs itself",
         [DQNAgentSplit(f'DQNs_{i}') for i in range(3)]),

        ("DQN shared-backbone vs itself",
         [DQNAgentShared(f'DQNh_{i}') for i in range(3)]),

        ("PPO vs PPO vs PPO",
         [PPOAgent(f'PPO_{i}') for i in range(3)]),

        # --- Three-way head-to-head ---
        ("QTable vs DQN vs PPO",
         [QTableAgent('QT'), DQNAgent('DQN'), PPOAgent('PPO')]),

        # --- Each agent vs 2 randoms (skill ceiling test) ---
        ("QTable vs 2 random",
         [QTableAgent('QT'), RandomAgent('R1'), RandomAgent('R2')]),

        ("DQN vs 2 random",
         [DQNAgent('DQN'), RandomAgent('R1'), RandomAgent('R2')]),

        ("PPO vs 2 random",
         [PPOAgent('PPO'), RandomAgent('R1'), RandomAgent('R2')]),
    ],
    train_episodes=1000,
    eval_episodes=200,
)
