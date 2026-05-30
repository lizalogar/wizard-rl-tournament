"""
visualize_training.py — train all three DQN variants and plot reward over time.

Run from the project root:
    python visualize_training.py
"""

import sys
import io
import numpy as np
import matplotlib.pyplot as plt

from common.game import WizardGame
from agents.agent_dqn import DQNAgent, DQNAgentSplit, DQNAgentShared

TRAIN_EPISODES = 2000
SMOOTH_WINDOW  = 80


def play_silent(game):
    sys.stdout = io.StringIO()
    try:
        return game.play_episode()
    finally:
        sys.stdout = sys.__stdout__


def train(AgentClass, label, n_episodes=TRAIN_EPISODES):
    agents = [AgentClass(f'{label}_{i}') for i in range(3)]
    print(f"Training {label} ({n_episodes} episodes)...", end='', flush=True)
    for ep in range(n_episodes):
        play_silent(WizardGame(agents))
        if (ep + 1) % 500 == 0:
            print(f' {ep+1}', end='', flush=True)
    print()
    return agents[0]


def smooth(values, window):
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode='valid')


baseline = train(DQNAgent,       'Baseline')
split    = train(DQNAgentSplit,  'Split')
shared   = train(DQNAgentShared, 'Shared')

plt.figure(figsize=(9, 5))
plt.plot(smooth(baseline.history['reward'], SMOOTH_WINDOW), label='Baseline (all 17 features)', color='#2196F3')
plt.plot(smooth(split.history['reward'],    SMOOTH_WINDOW), label='Split (5 / 12 features)',    color='#4CAF50')
plt.plot(smooth(shared.history['reward'],   SMOOTH_WINDOW), label='Shared backbone',             color='#FF9800')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
plt.title(f'DQN Variant Learning Curves  ({TRAIN_EPISODES} episodes, 3-agent self-play)')
plt.xlabel('Episode')
plt.ylabel('Score (smoothed)')
plt.legend()
plt.tight_layout()
plt.savefig('dqn_learning_curves.png', dpi=150)
print("Saved → dqn_learning_curves.png")
plt.show()
