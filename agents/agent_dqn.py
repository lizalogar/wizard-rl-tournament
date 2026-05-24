"""
DQN Agent — Deep Q-Network with experience replay and target network.

Implements proper TD-learning (Bellman equation):
  target = r + γ · max Q_target(s') · (1 − done)

The original skeleton used raw rewards as training targets, which is more
like supervised regression than reinforcement learning.  This version stores
full (s, a, r, s', done) transitions so the target network can bootstrap
future Q-values — the core idea behind DQN (Mnih et al., 2015).

Architecture
------------
  bid_net   12 → hidden → hidden → MAX_BID outputs
  play_net  12 → hidden → hidden → NUM_CARD_TYPES outputs

Each network has a frozen target copy synced every target_update_freq episodes.

Transition tracking
-------------------
  Bid  — terminal transition: (bid_obs, action, round_reward, zeros, done=True)
          The bid commits to a round outcome; round score is the full return.
  Play — sequential: (obs_t, action_t, trick_reward, obs_t+1, done)
          done=False between tricks, done=True on the last trick of a round.
          obs_t+1 is captured from the next call to act() in the play phase.
"""

import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from common.base_agent import (BaseAgent, OBS_SIZE,
                                OBS_PHASE, NUM_CARD_TYPES, MAX_BID)

_ZEROS = np.zeros(OBS_SIZE, dtype=np.float32)   # sentinel for terminal next_obs


# ------------------------------------------------------------------
# Network
# ------------------------------------------------------------------

class QNetwork(nn.Module):
    """Two hidden layers with ReLU; outputs one Q-value per action."""
    def __init__(self, input_size, output_size, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------
# Replay buffer  (full 5-tuple for TD learning)
# ------------------------------------------------------------------

class ReplayBuffer:
    """
    Stores (obs, action, reward, next_obs, done) tuples.

    The done flag lets the training step mask out future Q-values on
    terminal transitions (Bellman: target = r  when done, r + γ·V(s') otherwise).
    """
    def __init__(self, capacity=10_000):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action_idx, reward, next_obs, done):
        self.buf.append((obs, action_idx, reward, next_obs, float(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buf, min(batch_size, len(self.buf)))
        obs, a, r, next_obs, done = zip(*batch)
        return (
            torch.FloatTensor(np.array(obs)),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(np.array(next_obs)),
            torch.FloatTensor(done),
        )

    def __len__(self):
        return len(self.buf)


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------

class DQNAgent(BaseAgent):
    """
    DQN agent with separate bid/play networks, target networks, and
    proper (s, a, r, s', done) experience replay.
    """

    def __init__(self, name,
                 lr=0.001,
                 gamma=0.9,
                 epsilon=0.9,
                 epsilon_decay=0.997,
                 epsilon_min=0.05,
                 batch_size=64,
                 buffer_capacity=10_000,
                 target_update_freq=50,
                 hidden_size=64):
        super().__init__(name)

        self.gamma              = gamma
        self.epsilon            = epsilon
        self.epsilon_decay      = epsilon_decay
        self.epsilon_min        = epsilon_min
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self._episodes_done     = 0

        # Bid networks  (12 → MAX_BID)
        self.bid_net    = QNetwork(OBS_SIZE, MAX_BID, hidden_size)
        self.bid_target = QNetwork(OBS_SIZE, MAX_BID, hidden_size)
        self.bid_target.load_state_dict(self.bid_net.state_dict())
        self.bid_opt    = optim.Adam(self.bid_net.parameters(), lr=lr)

        # Play networks (12 → NUM_CARD_TYPES)
        self.play_net    = QNetwork(OBS_SIZE, NUM_CARD_TYPES, hidden_size)
        self.play_target = QNetwork(OBS_SIZE, NUM_CARD_TYPES, hidden_size)
        self.play_target.load_state_dict(self.play_net.state_dict())
        self.play_opt    = optim.Adam(self.play_net.parameters(), lr=lr)

        # Replay buffers
        self.bid_buf  = ReplayBuffer(buffer_capacity)
        self.play_buf = ReplayBuffer(buffer_capacity)

        # Pending transition state machine
        #   _bid_pending         (obs, action)         — awaits on_round_end reward
        #   _play_current        (obs, action)         — awaits on_trick_result reward
        #   _play_pending_reward (obs, action, reward) — awaits next obs from act()
        self._bid_pending         = None
        self._play_current        = None
        self._play_pending_reward = None

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray, valid_actions: list) -> int:
        phase = obs[OBS_PHASE]

        if phase < 0.5:
            # ----- Bidding -----
            # End of previous round: flush the last play transition as terminal.
            self._push_pending_play(next_obs=_ZEROS, done=True)

            if random.random() < self.epsilon:
                action = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    q = self.bid_net(torch.FloatTensor(obs)).numpy()
                for b in range(MAX_BID):
                    if b not in valid_actions:
                        q[b] = -1e9
                action = int(q.argmax())

            self._bid_pending  = (obs.copy(), action)
            self._play_current = None

        else:
            # ----- Playing -----
            # We now have obs_t+1 for whatever was pending from last trick.
            self._push_pending_play(next_obs=obs, done=False)

            if random.random() < self.epsilon:
                action = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    q = self.play_net(torch.FloatTensor(obs)).numpy()
                for i in range(NUM_CARD_TYPES):
                    if i not in valid_actions:
                        q[i] = -1e9
                action = int(q.argmax())

            self._play_current = (obs.copy(), action)

        return action

    def _push_pending_play(self, next_obs, done):
        """Complete and push a play transition once next_obs is known."""
        if self._play_pending_reward is not None:
            obs, action, reward = self._play_pending_reward
            self.play_buf.push(obs, action, reward, next_obs, done)
            self._play_pending_reward = None

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_trick_result(self, won_trick: bool, trick_reward: float = 0.0):
        """Pair the trick reward with the most recent play action."""
        if self._play_current is None:
            return
        obs, action = self._play_current
        self._play_pending_reward = (obs, action, trick_reward)
        self._play_current = None

    def on_round_end(self, reward: float):
        """Bid is a terminal decision — round score is its full return."""
        if self._bid_pending is not None:
            obs, action = self._bid_pending
            self.bid_buf.push(obs, action, reward, _ZEROS, True)
            self._bid_pending = None

    def on_episode_end(self):
        # Flush any dangling play transition from the last trick.
        self._push_pending_play(next_obs=_ZEROS, done=True)

        self._train_step(self.bid_net,  self.bid_target,  self.bid_opt,  self.bid_buf)
        self._train_step(self.play_net, self.play_target, self.play_opt, self.play_buf)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self._episodes_done += 1

        if self._episodes_done % self.target_update_freq == 0:
            self.bid_target.load_state_dict(self.bid_net.state_dict())
            self.play_target.load_state_dict(self.play_net.state_dict())

    # ------------------------------------------------------------------
    # Training step  (Bellman target with target network)
    # ------------------------------------------------------------------

    def _train_step(self, net, target_net, opt, buf):
        if len(buf) < self.batch_size:
            return

        obs, actions, rewards, next_obs, dones = buf.sample(self.batch_size)

        with torch.no_grad():
            next_q  = target_net(next_obs).max(1)[0]
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        q_pred = net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss   = nn.MSELoss()(q_pred, targets)

        opt.zero_grad()
        loss.backward()
        opt.step()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            'bid_net':  self.bid_net.state_dict(),
            'play_net': self.play_net.state_dict(),
            'epsilon':  self.epsilon,
        }, path)
        print(f"[DQNAgent] saved to {path}")

    def load(self, path: str):
        data = torch.load(path, weights_only=True)
        self.bid_net.load_state_dict(data['bid_net'])
        self.play_net.load_state_dict(data['play_net'])
        self.bid_target.load_state_dict(data['bid_net'])
        self.play_target.load_state_dict(data['play_net'])
        self.epsilon = data['epsilon']
        print(f"[DQNAgent] loaded from {path} (epsilon={self.epsilon:.3f})")
