"""
visualize_training.py — train all three DQN variants and plot learning curves.

Run from the project root:
    python visualize_training.py

Produces dqn_learning_curves.png with four subplots:
  - Episode reward (smoothed)   — is the agent scoring better over time?
  - Total loss (smoothed)       — is the network converging?
  - Epsilon decay               — how exploration decreases over training
  - Reward distribution (box)   — spread of scores in early vs late training
"""

import sys
import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from common.game import WizardGame
from agents.agent_dqn import DQNAgent, DQNAgentSplit, DQNAgentShared

TRAIN_EPISODES = 2000
SMOOTH_WINDOW  = 80    # rolling average window for reward and loss plots


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def play_silent(game):
    sys.stdout = io.StringIO()
    try:
        return game.play_episode()
    finally:
        sys.stdout = sys.__stdout__


def train(AgentClass, label, n_episodes=TRAIN_EPISODES):
    agents = [AgentClass(f'{label}_{i}') for i in range(3)]
    print(f"  Training {label} ({n_episodes} episodes)...", end='', flush=True)
    for ep in range(n_episodes):
        play_silent(WizardGame(agents))
        if (ep + 1) % 500 == 0:
            print(f' {ep+1}', end='', flush=True)
    print()
    return agents[0]   # return agent 0 — it holds the history


def smooth(values, window):
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------

print("=" * 50)
print("  DQN Variant Training — Learning Curve Visualizer")
print("=" * 50)
print()

baseline = train(DQNAgent,       'Baseline')
split    = train(DQNAgentSplit,  'Split')
shared   = train(DQNAgentShared, 'Shared')

variants = [
    ('Baseline (all 12 features)', baseline, '#2196F3'),
    ('Split input (4 / 7 features)', split,  '#4CAF50'),
    ('Shared backbone',             shared,  '#FF9800'),
]

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------

fig = plt.figure(figsize=(14, 10))
fig.suptitle(
    f'DQN Variant Learning Curves  ({TRAIN_EPISODES} episodes, 3-agent self-play)',
    fontsize=13, fontweight='bold'
)
gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

# --- 1. Episode reward (smoothed) ---
ax1 = fig.add_subplot(gs[0, 0])
for label, agent, color in variants:
    r = smooth(agent.history['reward'], SMOOTH_WINDOW)
    ax1.plot(r, label=label, color=color, linewidth=1.5)
ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax1.set_title('Episode Reward (smoothed)')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Score')
ax1.legend(fontsize=8)

# --- 2. Total loss (smoothed) ---
ax2 = fig.add_subplot(gs[0, 1])
for label, agent, color in variants:
    # exclude zeros (episodes before buffer was full)
    loss = [v for v in agent.history['loss'] if v > 0]
    ax2.plot(smooth(loss, SMOOTH_WINDOW), label=label, color=color, linewidth=1.5)
ax2.set_title('Training Loss (smoothed)')
ax2.set_xlabel('Episode (after buffer warm-up)')
ax2.set_ylabel('MSE Loss')
ax2.legend(fontsize=8)

# --- 3. Epsilon decay ---
ax3 = fig.add_subplot(gs[1, 0])
for label, agent, color in variants:
    ax3.plot(agent.history['epsilon'], label=label, color=color, linewidth=1.5)
ax3.set_title('Epsilon Decay  (exploration → exploitation)')
ax3.set_xlabel('Episode')
ax3.set_ylabel('Epsilon')
ax3.legend(fontsize=8)

# --- 4. Reward distribution: early vs late ---
ax4 = fig.add_subplot(gs[1, 1])
split_at = TRAIN_EPISODES // 2
positions = [1, 2, 3, 5, 6, 7]
colors_box = []
data_box   = []
tick_labels = []

for i, (label, agent, color) in enumerate(variants):
    r = agent.history['reward']
    data_box.append(r[:split_at])    # first half
    data_box.append(r[split_at:])    # second half
    colors_box.extend([color, color])
    short = label.split(' ')[0]
    tick_labels.extend([f'{short}\nearly', f'{short}\nlate'])

bp = ax4.boxplot(data_box, positions=positions, patch_artist=True,
                 widths=0.6, showfliers=False,
                 medianprops=dict(color='black', linewidth=1.5))

for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax4.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax4.set_xticks(positions)
ax4.set_xticklabels(tick_labels, fontsize=7)
ax4.set_title('Score Distribution: Early vs Late Training')
ax4.set_ylabel('Score per Episode')

out_path = 'dqn_learning_curves.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved → {out_path}")
plt.show()
