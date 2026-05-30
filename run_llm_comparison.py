"""
run_llm_comparison.py
=====================
Can Claude (the LLM) play Wizard better than the trained RL models?

This script:
  1. Trains QTable, DQN, and PPO agents via 3-agent self-play.
  2. Runs a 7-way evaluate_vs_self() (epsilon=0, greedy) for EVAL_EPISODES
     games with all agents: QT, DQN, PPO, Random, Heuristic, DT, Claude.
  3. Saves results to  llm_comparison_results.json.

Usage
-----
    python run_llm_comparison.py

Pre-trained weights in models/ are loaded automatically, skipping training.
"""

import os
import json
import time

from agents.agent_qtable       import QTableAgent
from agents.agent_dqn          import DQNAgent
from agents.agent_ppo          import PPOAgent
from agents.random_agent       import RandomAgent
from agents.heuristic_agent    import HeuristicAgent
from agents.heuristic_dt_agent import HeuristicDTAgent
from agents.agent_claude       import ClaudeAgent
from common.tournament         import train_agents, evaluate_vs_self

# ──────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────

MODELS_DIR     = 'models'
TRAIN_EPISODES = 5000    # self-play training episodes for learnable agents
EVAL_EPISODES  = 1000    # evaluation games (no exploration)
RESULTS_FILE   = 'llm_comparison_results.json'

os.makedirs(MODELS_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────
# Helper — train or load one agent type
# ──────────────────────────────────────────────────────────────────

def get_trained_agent(AgentClass, name):
    """
    Return a trained agent.
    Loads from disk if the .pt file exists; otherwise trains from scratch
    using 3-agent self-play, saves the result, and returns agent[0].
    """
    agent = AgentClass(name)
    path  = os.path.join(MODELS_DIR, f'{name}.pt')

    if os.path.exists(path):
        print(f"  [{name}] Loading pre-trained weights from {path} ...")
        agent.load(path)
        agent.epsilon = 0.0
    else:
        print(f"  [{name}] No saved weights — training {TRAIN_EPISODES} episodes ...",
              end='', flush=True)
        t0 = time.time()
        # 3-agent self-play: agent participates as player 0; two fresh
        # copies act as opponents and update independently.
        trio = [AgentClass(f'{name}_{i}') for i in range(3)]
        train_agents(trio, TRAIN_EPISODES)

        # Keep only the first copy as our representative
        trained = trio[0]
        trained.name    = name       # rename to canonical name
        trained.epsilon = 0.0        # greedy for evaluation
        trained.save(path)
        agent = trained
        print(f" done ({time.time()-t0:.0f}s)")

    return agent


# ──────────────────────────────────────────────────────────────────
# Prepare all agents
# ──────────────────────────────────────────────────────────────────

print("\n" + "="*68)
print("  LLM vs RL Comparison — Wizard Card Game")
print("="*68)

qt_agent  = get_trained_agent(QTableAgent, 'QT')
dqn_agent = get_trained_agent(DQNAgent,    'DQN')
ppo_agent = get_trained_agent(PPOAgent,    'PPO')
ppo_agent.epsilon = 0.0   # PPO also uses epsilon flag

all_agents = [
    qt_agent,
    dqn_agent,
    ppo_agent,
    RandomAgent('Random'),
    HeuristicAgent('H'),
    HeuristicDTAgent('DT'),
    ClaudeAgent('Claude'),
]

# Freeze exploration on all learnable agents
for a in all_agents:
    if hasattr(a, 'epsilon'):
        a.epsilon = 0.0


# ──────────────────────────────────────────────────────────────────
# 7-way head-to-head evaluation
# ──────────────────────────────────────────────────────────────────

print(f"\n{'-'*68}")
print(f"  Running {EVAL_EPISODES}-game evaluation (7 agents, greedy play)")
print(f"  Players: {[a.name for a in all_agents]}")
print(f"{'-'*68}")

t0         = time.time()
avg_scores = evaluate_vs_self(all_agents, num_episodes=EVAL_EPISODES)
elapsed    = time.time() - t0


# ──────────────────────────────────────────────────────────────────
# Results table
# ──────────────────────────────────────────────────────────────────

results = sorted(
    zip([a.name for a in all_agents], avg_scores),
    key=lambda x: x[1],
    reverse=True,
)

TAGS = {
    'Claude': '[LLM - this agent]',
    'H':      '[hand-coded heuristic]',
    'DT':     '[decision-tree heuristic]',
    'Random': '[random baseline]',
    'QT':     f'[Q-table, {TRAIN_EPISODES} ep]',
    'DQN':    f'[DQN, {TRAIN_EPISODES} ep]',
    'PPO':    f'[PPO, {TRAIN_EPISODES} ep]',
}

print(f"\n{'='*68}")
print(f"  FINAL RANKINGS — avg score/game over {EVAL_EPISODES} episodes")
print(f"{'='*68}")
print(f"  {'Rank':<5} {'Agent':<12} {'Avg Score':>10}  {'Notes'}")
print(f"  {'-'*5} {'-'*12} {'-'*10}  {'-'*26}")

for rank, (name, score) in enumerate(results, 1):
    marker = '  <<<' if name == 'Claude' else ''
    print(f"  #{rank:<4} {name:<12} {score:>10.1f}  {TAGS.get(name,'')}{marker}")

print(f"\n  Evaluation time: {elapsed:.1f}s for {EVAL_EPISODES} games")


# ──────────────────────────────────────────────────────────────────
# Save JSON
# ──────────────────────────────────────────────────────────────────

output = {
    'experiment':      'LLM (Claude) vs RL Comparison — Wizard card game',
    'eval_episodes':   EVAL_EPISODES,
    'train_episodes':  TRAIN_EPISODES,
    'num_players':     len(all_agents),
    'rankings': [
        {'rank': rank, 'agent': name, 'avg_score': round(score, 2),
         'type': TAGS.get(name, '')}
        for rank, (name, score) in enumerate(results, 1)
    ],
    'raw_scores': {
        name: round(score, 2)
        for name, score in zip([a.name for a in all_agents], avg_scores)
    },
}

with open(RESULTS_FILE, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n  Results saved -> {RESULTS_FILE}")
print("="*68 + "\n")
