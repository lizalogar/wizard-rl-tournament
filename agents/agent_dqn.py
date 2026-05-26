"""
DQN Agent — three architectural variants for comparison.

  DQNAgent        baseline  both networks receive all 12 obs features
  DQNAgentSplit   option 1  each network receives only its relevant features
  DQNAgentShared  option 2  shared backbone (12→64→64) with two output heads

Run all three in run_tournament.py to see which architecture learns best.
Each agent records self.history with per-episode reward, loss, and epsilon
for use in visualize_training.py.
"""

import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from common.base_agent import (BaseAgent, OBS_SIZE,
                                OBS_PHASE, NUM_CARD_TYPES, MAX_BID)


# ------------------------------------------------------------------
# Feature subsets used by DQNAgentSplit
# ------------------------------------------------------------------
# Only the features that are actually meaningful for each decision.

# those numbers are the indices of the features in the full obs vector that are relevant for each network
BID_FEATURES  = [0, 6, 7, 8]              # round, wizards, high, trump in hand
PLAY_FEATURES = [2, 3, 4, 5, 9, 10, 11]  # bid, tricks_won, needed, position,
                                           # have_wizard / have_jester / have_trump

_ZEROS_FULL = np.zeros(OBS_SIZE,            dtype=np.float32)  # terminal s' for full obs
_ZEROS_BID  = np.zeros(len(BID_FEATURES),  dtype=np.float32)  # terminal s' for bid subset
_ZEROS_PLAY = np.zeros(len(PLAY_FEATURES), dtype=np.float32)  # terminal s' for play subset


# ------------------------------------------------------------------
# Neural networks
# ------------------------------------------------------------------
# Bid network:  12 inputs → 11 outputs (one Q-value per possible bid 0–10)
# Play network: 12 inputs → 5 outputs  (one Q-value per card type)

class QNetwork(nn.Module):
    """Standard two-hidden-layer network; outputs one Q-value per action."""
    def __init__(self, input_size, output_size, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)
# forward just returns the output — it does not train the network. It's a pure forward pass: input goes in, Q-values come out.

class SharedQNetwork(nn.Module):
    """
    Shared backbone with two output heads.

    obs (12) → backbone (12 → 64 → 64) ─┬→ bid_head  (64 → MAX_BID)
                                          └→ play_head (64 → NUM_CARD_TYPES)

    Both heads share the same hidden representation, so knowledge about
    trick-winning can inform bidding and vice versa.
    """
    def __init__(self, input_size, hidden_size, bid_outputs, play_outputs):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.bid_head  = nn.Linear(hidden_size, bid_outputs)
        self.play_head = nn.Linear(hidden_size, play_outputs)

    def forward_bid(self, x):
        return self.bid_head(self.backbone(x))

    def forward_play(self, x):
        return self.play_head(self.backbone(x))


# ------------------------------------------------------------------
# Replay buffer
# ------------------------------------------------------------------
# A buffer is just a storage container — in this case a list of
# past experiences the agent has collected while playing games.

# Without it, the agent would learn from each experience the
# moment it happens and then throw it away.

# The replay buffer fixes this by saving experiences
# and training on random mixtures of them later:

class ReplayBuffer:
    """
    Stores (obs, action, reward, next_obs, done) tuples.

    done=True means the episode/round ended after this step, so the
    Bellman target is just r (no future Q to bootstrap from).
    """
    def __init__(self, capacity=10_000):
        self.buf = deque(maxlen=capacity)
        # deque is a Python list with a maximum size. Once it hits 10,000 entries,
        # adding a new one automatically drops the oldest. No manual cleanup needed.

    def push(self, obs, action_idx, reward, next_obs, done):
        self.buf.append((obs, action_idx, reward, next_obs, float(done)))

    def sample(self, batch_size):
        # sample — pulls a random mini-batch for training:
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


# ==================================================================
# VARIANT 1 — DQNAgent
# Baseline: both networks receive all 12 obs features.
# ==================================================================

class DQNAgent(BaseAgent):

    def __init__(self, name,
                 lr=0.001, gamma=0.9,
                 epsilon=0.9, epsilon_decay=0.997, epsilon_min=0.05,
                 batch_size=64, buffer_capacity=10_000,
                 target_update_freq=50, hidden_size=64):
        super().__init__(name)
        self.gamma              = gamma
        self.epsilon            = epsilon
        self.epsilon_decay      = epsilon_decay
        self.epsilon_min        = epsilon_min
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self._episodes_done     = 0

        self.bid_net    = QNetwork(OBS_SIZE, MAX_BID, hidden_size)
        self.bid_target = QNetwork(OBS_SIZE, MAX_BID, hidden_size)
        self.bid_target.load_state_dict(self.bid_net.state_dict())
        self.bid_opt    = optim.Adam(self.bid_net.parameters(), lr=lr)

        self.play_net    = QNetwork(OBS_SIZE, NUM_CARD_TYPES, hidden_size)
        self.play_target = QNetwork(OBS_SIZE, NUM_CARD_TYPES, hidden_size)
        self.play_target.load_state_dict(self.play_net.state_dict())
        self.play_opt    = optim.Adam(self.play_net.parameters(), lr=lr)

        self.bid_buf  = ReplayBuffer(buffer_capacity)
        self.play_buf = ReplayBuffer(buffer_capacity)

        self._bid_pending         = None
        self._play_current        = None
        self._play_pending_reward = None

        # metric tracking — one entry appended per episode in on_episode_end
        self.history = {'reward': [], 'loss': [], 'epsilon': []}
        self._episode_reward = 0.0

    def act(self, obs, valid_actions):
        phase = obs[OBS_PHASE]
        if phase < 0.5:
            self._push_pending_play(_ZEROS_FULL, done=True)
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
            self._push_pending_play(obs, done=False)
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
        if self._play_pending_reward is not None:
            obs, action, reward = self._play_pending_reward
            self.play_buf.push(obs, action, reward, next_obs, done)
            self._play_pending_reward = None

    def on_trick_result(self, _, trick_reward=0.0):
        if self._play_current is None:
            return
        obs, action = self._play_current
        self._play_pending_reward = (obs, action, trick_reward)
        self._play_current = None

    def on_round_end(self, reward):
        self._episode_reward += reward
        if self._bid_pending is not None:
            obs, action = self._bid_pending
            self.bid_buf.push(obs, action, reward, _ZEROS_FULL, True)
            self._bid_pending = None

    def on_episode_end(self):
        self._push_pending_play(_ZEROS_FULL, done=True)
        bid_loss  = self._train_step(self.bid_net,  self.bid_target,  self.bid_opt,  self.bid_buf)
        play_loss = self._train_step(self.play_net, self.play_target, self.play_opt, self.play_buf)

        self.history['reward'].append(self._episode_reward)
        self.history['loss'].append(bid_loss + play_loss)
        self.history['epsilon'].append(self.epsilon)
        self._episode_reward = 0.0

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self._episodes_done += 1
        if self._episodes_done % self.target_update_freq == 0:
            self.bid_target.load_state_dict(self.bid_net.state_dict())
            self.play_target.load_state_dict(self.play_net.state_dict())

    def _train_step(self, net, target_net, opt, buf):
        if len(buf) < self.batch_size:
            return 0.0
        obs, actions, rewards, next_obs, dones = buf.sample(self.batch_size)
        with torch.no_grad():
            next_q  = target_net(next_obs).max(1)[0]
            targets = rewards + self.gamma * next_q * (1.0 - dones)
        q_pred = net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss   = nn.MSELoss()(q_pred, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()

    def save(self, path):
        torch.save({'bid_net': self.bid_net.state_dict(),
                    'play_net': self.play_net.state_dict(),
                    'epsilon': self.epsilon}, path)

    def load(self, path):
        data = torch.load(path, weights_only=True)
        self.bid_net.load_state_dict(data['bid_net'])
        self.play_net.load_state_dict(data['play_net'])
        self.bid_target.load_state_dict(data['bid_net'])
        self.play_target.load_state_dict(data['play_net'])
        self.epsilon = data['epsilon']


# ==================================================================
# VARIANT 2 — DQNAgentSplit
# Option 1: each network receives only its relevant feature subset.
#   bid_net  → 4 inputs  (BID_FEATURES)
#   play_net → 7 inputs  (PLAY_FEATURES)
# ==================================================================

class DQNAgentSplit(BaseAgent):

    def __init__(self, name,
                 lr=0.001, gamma=0.9,
                 epsilon=0.9, epsilon_decay=0.997, epsilon_min=0.05,
                 batch_size=64, buffer_capacity=10_000,
                 target_update_freq=50, hidden_size=64):
        super().__init__(name)
        self.gamma              = gamma
        self.epsilon            = epsilon
        self.epsilon_decay      = epsilon_decay
        self.epsilon_min        = epsilon_min
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self._episodes_done     = 0

        # Difference 1 — smaller network inputs:
        bid_in  = len(BID_FEATURES)
        play_in = len(PLAY_FEATURES)

        self.bid_net    = QNetwork(bid_in, MAX_BID, hidden_size)
        self.bid_target = QNetwork(bid_in, MAX_BID, hidden_size)
        self.bid_target.load_state_dict(self.bid_net.state_dict())
        self.bid_opt    = optim.Adam(self.bid_net.parameters(), lr=lr)

        self.play_net    = QNetwork(play_in, NUM_CARD_TYPES, hidden_size)
        self.play_target = QNetwork(play_in, NUM_CARD_TYPES, hidden_size)
        self.play_target.load_state_dict(self.play_net.state_dict())
        self.play_opt    = optim.Adam(self.play_net.parameters(), lr=lr)

        self.bid_buf  = ReplayBuffer(buffer_capacity)
        self.play_buf = ReplayBuffer(buffer_capacity)

        self._bid_pending         = None
        self._play_current        = None
        self._play_pending_reward = None

        self.history = {'reward': [], 'loss': [], 'epsilon': []}
        self._episode_reward = 0.0

    def act(self, obs, valid_actions):
        phase = obs[OBS_PHASE]
        if phase < 0.5:
            self._push_pending_play(None, done=True)
            bid_obs = obs[BID_FEATURES]
            if random.random() < self.epsilon:
                action = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    q = self.bid_net(torch.FloatTensor(bid_obs)).numpy()
                for b in range(MAX_BID):
                    if b not in valid_actions:
                        q[b] = -1e9
                action = int(q.argmax())
            self._bid_pending  = (bid_obs.copy(), action)
            self._play_current = None
        else:
            play_obs = obs[PLAY_FEATURES]
            self._push_pending_play(play_obs, done=False)
            if random.random() < self.epsilon:
                action = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    q = self.play_net(torch.FloatTensor(play_obs)).numpy()
                for i in range(NUM_CARD_TYPES):
                    if i not in valid_actions:
                        q[i] = -1e9
                action = int(q.argmax())
            self._play_current = (play_obs.copy(), action)
        return action

    def _push_pending_play(self, next_obs, done):
        if self._play_pending_reward is not None:
            obs, action, reward = self._play_pending_reward
            nxt = _ZEROS_PLAY if done else next_obs
            self.play_buf.push(obs, action, reward, nxt, done)
            self._play_pending_reward = None

    def on_trick_result(self, _, trick_reward=0.0):
        if self._play_current is None:
            return
        obs, action = self._play_current
        self._play_pending_reward = (obs, action, trick_reward)
        self._play_current = None

    def on_round_end(self, reward):
        self._episode_reward += reward
        if self._bid_pending is not None:
            obs, action = self._bid_pending
            self.bid_buf.push(obs, action, reward, _ZEROS_BID, True)
            self._bid_pending = None

    def on_episode_end(self):
        self._push_pending_play(None, done=True)
        bid_loss  = self._train_step(self.bid_net,  self.bid_target,  self.bid_opt,  self.bid_buf)
        play_loss = self._train_step(self.play_net, self.play_target, self.play_opt, self.play_buf)

        self.history['reward'].append(self._episode_reward)
        self.history['loss'].append(bid_loss + play_loss)
        self.history['epsilon'].append(self.epsilon)
        self._episode_reward = 0.0

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self._episodes_done += 1
        if self._episodes_done % self.target_update_freq == 0:
            self.bid_target.load_state_dict(self.bid_net.state_dict())
            self.play_target.load_state_dict(self.play_net.state_dict())

    def _train_step(self, net, target_net, opt, buf):
        if len(buf) < self.batch_size:
            return 0.0
        obs, actions, rewards, next_obs, dones = buf.sample(self.batch_size)
        with torch.no_grad():
            next_q  = target_net(next_obs).max(1)[0]
            targets = rewards + self.gamma * next_q * (1.0 - dones)
        q_pred = net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss   = nn.MSELoss()(q_pred, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()

    def save(self, path):
        torch.save({'bid_net': self.bid_net.state_dict(),
                    'play_net': self.play_net.state_dict(),
                    'epsilon': self.epsilon}, path)

    def load(self, path):
        data = torch.load(path, weights_only=True)
        self.bid_net.load_state_dict(data['bid_net'])
        self.play_net.load_state_dict(data['play_net'])
        self.bid_target.load_state_dict(data['bid_net'])
        self.play_target.load_state_dict(data['play_net'])
        self.epsilon = data['epsilon']


# ==================================================================
# VARIANT 3 — DQNAgentShared
# Option 2: single network with a shared backbone and two output heads.
# Both heads are trained in one backward pass, so the backbone learns
# a representation useful for both bidding and playing simultaneously.
# ==================================================================

class DQNAgentShared(BaseAgent):

    def __init__(self, name,
                 lr=0.001, gamma=0.9,
                 epsilon=0.9, epsilon_decay=0.997, epsilon_min=0.05,
                 batch_size=64, buffer_capacity=10_000,
                 target_update_freq=50, hidden_size=64):
        super().__init__(name)
        self.gamma              = gamma
        self.epsilon            = epsilon
        self.epsilon_decay      = epsilon_decay
        self.epsilon_min        = epsilon_min
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self._episodes_done     = 0

        self.net    = SharedQNetwork(OBS_SIZE, hidden_size, MAX_BID, NUM_CARD_TYPES)
        self.target = SharedQNetwork(OBS_SIZE, hidden_size, MAX_BID, NUM_CARD_TYPES)
        self.target.load_state_dict(self.net.state_dict())
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

        self.bid_buf  = ReplayBuffer(buffer_capacity)
        self.play_buf = ReplayBuffer(buffer_capacity)

        self._bid_pending         = None
        self._play_current        = None
        self._play_pending_reward = None

        self.history = {'reward': [], 'loss': [], 'epsilon': []}
        self._episode_reward = 0.0

    def act(self, obs, valid_actions):
        phase = obs[OBS_PHASE]
        if phase < 0.5:
            self._push_pending_play(_ZEROS_FULL, done=True)
            if random.random() < self.epsilon:
                action = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    q = self.net.forward_bid(torch.FloatTensor(obs)).numpy()
                for b in range(MAX_BID):
                    if b not in valid_actions:
                        q[b] = -1e9
                action = int(q.argmax())
            self._bid_pending  = (obs.copy(), action)
            self._play_current = None
        else:
            self._push_pending_play(obs, done=False)
            if random.random() < self.epsilon:
                action = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    q = self.net.forward_play(torch.FloatTensor(obs)).numpy()
                for i in range(NUM_CARD_TYPES):
                    if i not in valid_actions:
                        q[i] = -1e9
                action = int(q.argmax())
            self._play_current = (obs.copy(), action)
        return action

    def _push_pending_play(self, next_obs, done):
        if self._play_pending_reward is not None:
            obs, action, reward = self._play_pending_reward
            self.play_buf.push(obs, action, reward, next_obs, done)
            self._play_pending_reward = None

    def on_trick_result(self, _, trick_reward=0.0):
        if self._play_current is None:
            return
        obs, action = self._play_current
        self._play_pending_reward = (obs, action, trick_reward)
        self._play_current = None

    def on_round_end(self, reward):
        self._episode_reward += reward
        if self._bid_pending is not None:
            obs, action = self._bid_pending
            self.bid_buf.push(obs, action, reward, _ZEROS_FULL, True)
            self._bid_pending = None

    def on_episode_end(self):
        self._push_pending_play(_ZEROS_FULL, done=True)
        total_loss = self._train_step()

        self.history['reward'].append(self._episode_reward)
        self.history['loss'].append(total_loss)
        self.history['epsilon'].append(self.epsilon)
        self._episode_reward = 0.0

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self._episodes_done += 1
        if self._episodes_done % self.target_update_freq == 0:
            self.target.load_state_dict(self.net.state_dict())

    def _train_step(self):
        """Single backward pass — bid loss + play loss flow through the shared backbone."""
        total_loss = None

        if len(self.bid_buf) >= self.batch_size:
            obs, actions, rewards, next_obs, dones = self.bid_buf.sample(self.batch_size)
            with torch.no_grad():
                next_q  = self.target.forward_bid(next_obs).max(1)[0]
                targets = rewards + self.gamma * next_q * (1.0 - dones)
            q_pred     = self.net.forward_bid(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
            total_loss = nn.MSELoss()(q_pred, targets)

        if len(self.play_buf) >= self.batch_size:
            obs, actions, rewards, next_obs, dones = self.play_buf.sample(self.batch_size)
            with torch.no_grad():
                next_q  = self.target.forward_play(next_obs).max(1)[0]
                targets = rewards + self.gamma * next_q * (1.0 - dones)
            q_pred    = self.net.forward_play(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
            play_loss = nn.MSELoss()(q_pred, targets)
            total_loss = play_loss if total_loss is None else total_loss + play_loss

        if total_loss is not None:
            self.opt.zero_grad()
            total_loss.backward()
            self.opt.step()
            return total_loss.item()
        return 0.0

    def save(self, path):
        torch.save({'net': self.net.state_dict(), 'epsilon': self.epsilon}, path)

    def load(self, path):
        data = torch.load(path, weights_only=True)
        self.net.load_state_dict(data['net'])
        self.target.load_state_dict(data['net'])
        self.epsilon = data['epsilon']
