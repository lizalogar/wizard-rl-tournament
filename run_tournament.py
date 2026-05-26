"""
run_tournament.py  —  entry point for the shared project.

Run from the project root:
    python run_tournament.py

Set LOAD_PRETRAINED = True to skip training and load saved models from
the models/ folder instead — useful for competing pre-trained agents.
"""

import os
from agents.agent_qtable import QTableAgent
from agents.agent_dqn    import DQNAgent, DQNAgentSplit, DQNAgentShared
from agents.agent_ppo    import PPOAgent
from agents.random_agent import RandomAgent
from common.tournament   import run_tournament, grid_search, evaluate_vs_self

# ================================================================
# Config
# ================================================================

MODELS_DIR      = 'models' 
LOAD_PRETRAINED = False   # True = load saved weights, skip training
TRAIN_EPISODES  = 4000    # increased from 1000 — QTable needs more with larger dims
EVAL_EPISODES   = 200

os.makedirs(MODELS_DIR, exist_ok=True)


# ================================================================
# Optional: grid search to find best hyperparameters per agent
# ================================================================

# print("\n>>> QTableAgent grid search")
# qt_results = grid_search(
#     QTableAgent,
#     param_grid={
#         'alpha':         [0.05, 0.1, 0.2],
#         'gamma':         [0.8, 0.9],
#         'epsilon_decay': [0.995, 0.997, 0.999],
#     },
#     train_episodes=3000,
#     eval_episodes=200,
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
# Matchups
# ================================================================

matchups = [
    # --- Baselines ---
    ("Random x3",
     [RandomAgent(f'R{i}') for i in range(3)]),

    # --- Each agent type playing against itself (self-play) ---
    ("QTable vs QTable vs QTable",
     [QTableAgent(f'QT_{i}') for i in range(3)]),

    # ("DQN baseline vs itself",
    #  [DQNAgent(f'DQN_{i}') for i in range(3)]),

    # ("DQN split-input vs itself",
    #  [DQNAgentSplit(f'DQNs_{i}') for i in range(3)]),

    # ("DQN shared-backbone vs itself",
    #  [DQNAgentShared(f'DQNh_{i}') for i in range(3)]),

    # ("PPO vs PPO vs PPO",
    #  [PPOAgent(f'PPO_{i}') for i in range(3)]),

    # --- Three-way head-to-head ---
    ("QTable vs DQN vs PPO",
     [QTableAgent('QT'), DQNAgent('DQN'), PPOAgent('PPO')]),

    # --- Each agent vs 2 randoms (skill ceiling test) ---
    ("QTable vs 2 random",
     [QTableAgent('QT_rand'), RandomAgent('R1'), RandomAgent('R2')]),

    # ("DQN vs 2 random",
    #  [DQNAgent('DQN'), RandomAgent('R1'), RandomAgent('R2')]),

    # ("PPO vs 2 random",
    #  [PPOAgent('PPO'), RandomAgent('R1'), RandomAgent('R2')]),
]


# ================================================================
# Load pre-trained weights (if requested)
# ================================================================

if LOAD_PRETRAINED:
    print(f"Loading pre-trained models from '{MODELS_DIR}/' ...")
    for _, agents in matchups:
        for agent in agents:
            path = os.path.join(MODELS_DIR, f'{agent.name}.pt')
            if os.path.exists(path):
                agent.load(path)
    train_eps = 0   # skip training, go straight to evaluation
else:
    train_eps = TRAIN_EPISODES


# ================================================================
# Run tournament
# ================================================================

run_tournament(matchups, train_episodes=train_eps, eval_episodes=EVAL_EPISODES)


# ================================================================
# Save trained models
# ================================================================

if not LOAD_PRETRAINED:
    print(f"\nSaving trained models to '{MODELS_DIR}/' ...")
    for _, agents in matchups:
        for agent in agents:
            if not isinstance(agent, RandomAgent):
                agent.save(os.path.join(MODELS_DIR, f'{agent.name}.pt'))
    print("Done.")


#------------------------------
# FINAL RUN WITH SCORES
#------------------------------
print("\n" + "="*70)
print(" GRAND FINALE: Individual Agent Scores")
print("="*70)

finale_agents = [
    QTableAgent('QT'),
    DQNAgent('DQN'),
    PPOAgent('PPO'),
    # You can even throw your new decision tree agent in here!
    # HeuristicDTAgent('DT') 
]

# 2. Load their trained weights
for agent in finale_agents:
    if hasattr(agent, 'load'): # Random and Heuristic agents don't have weights to load
        try:
            agent.load(os.path.join(MODELS_DIR, f'{agent.name}.pt'))
        except FileNotFoundError:
            print(f"Warning: Could not find trained weights for {agent.name}")

# 3. We can only play 3 agents at a time in Wizard.
# Let's do a strict QT vs DQN vs PPO match.
matchup = [finale_agents[0], finale_agents[1], finale_agents[2]]
print(f"Playing 1000 evaluation games between: {[a.name for a in matchup]}...\n")

# evaluate_vs_self sets epsilon to 0 (greedy/evaluation mode) and returns a list of average scores
final_scores = evaluate_vs_self(matchup, num_episodes=1000)

for agent, score in zip(matchup, final_scores):
    print(f"{agent.name:>10}: {score:>6.1f} avg points/game")