"""
PPO Agent  —  YOUR FILE TO IMPLEMENT.

The engine always calls act(obs, valid_actions) and the three on_* hooks.
Do NOT touch any file in common/.

What is PPO?
------------
Proximal Policy Optimisation is a policy-gradient method — it directly
learns a probability distribution over actions (a "policy") rather than
Q-values.  It has two networks:

  policy  π(a | s)  — outputs a probability for each action given state s
  value   V(s)      — estimates the expected total reward from state s

Training loop (once per episode in on_episode_end):
  1. Collect all (obs, action, log_prob, reward) from the episode
  2. Compute advantages:  A = reward - V(s)
     (advantage > 0  →  action was better than expected, reinforce it)
  3. PPO "clipped" objective:
       r = π_new(a|s) / π_old(a|s)          (probability ratio)
       L = E[ min( r*A,  clip(r, 1-ε, 1+ε)*A ) ]
     The clip prevents the policy from updating too aggressively in one step,
     which stabilises training compared to vanilla policy gradient (REINFORCE).
  4. Value loss:  MSE( V(s),  reward )
  5. Gradient step on both networks.

Key difference from DQN
  DQN: deterministic (picks highest Q-value)
  PPO: stochastic (samples from learned probability distribution)
       → naturally explores without needing an explicit epsilon

Suggested improvements:
  - Generalised Advantage Estimation (GAE) for step 2
  - Entropy bonus to prevent policy collapse
  - Multiple update epochs per episode (K=4 is common)
  - Separate hidden sizes for policy and value networks
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from common.base_agent import (BaseAgent, OBS_SIZE, OBS_PHASE,
                                NUM_CARD_TYPES, MAX_BID)


# ------------------------------------------------------------------
# Networks
# ------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    """
    Maps obs -> raw logits (one per action).
    We apply softmax + masking externally so that invalid actions get
    probability 0 regardless of network output.
    """
    def __init__(self, input_size, output_size, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)   # raw logits


class ValueNetwork(nn.Module):
    """Maps obs -> scalar value estimate."""
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------

class PPOAgent(BaseAgent):
    """
    PPO agent with separate policy networks for bidding and playing
    and a shared value network.

    self.epsilon is kept only so the tournament's _freeze_epsilon()
    utility can switch the agent to greedy evaluation mode.
    """

    def __init__(self, name,
                 lr=0.001,
                 gamma=0.9,
                 clip_eps=0.2,     # PPO clipping range
                 hidden_size=64):
        super().__init__(name)
        self.gamma    = gamma
        self.clip_eps = clip_eps

        # epsilon=1 → stochastic (training),  epsilon=0 → greedy (evaluation)
        self.epsilon = 1.0

        # Separate policy nets for the two decision types
        self.bid_policy  = PolicyNetwork(OBS_SIZE, MAX_BID,        hidden_size)
        self.play_policy = PolicyNetwork(OBS_SIZE, NUM_CARD_TYPES, hidden_size)
        self.value_net   = ValueNetwork(OBS_SIZE, hidden_size)

        self.bid_opt   = optim.Adam(self.bid_policy.parameters(),  lr=lr)
        self.play_opt  = optim.Adam(self.play_policy.parameters(), lr=lr)
        self.value_opt = optim.Adam(self.value_net.parameters(),   lr=lr)

        # Trajectory collected over the episode
        # Each entry: (obs, action, log_prob, phase)
        # reward is filled in by on_round_end
        self._pending    = []   # transitions waiting for their reward
        self._trajectory = []   # (obs, action, log_prob, reward, phase)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray, valid_actions: list) -> int:
        obs_t = torch.FloatTensor(obs)
        phase = obs[OBS_PHASE]

        policy = self.bid_policy if phase < 0.5 else self.play_policy

        with torch.no_grad():
            logits = policy(obs_t)

        # Mask invalid actions: set their logits to -inf before softmax
        mask = torch.full((logits.shape[0],), float('-inf'))
        for a in valid_actions:
            mask[a] = 0.0
        logits = logits + mask

        probs = torch.softmax(logits, dim=-1)

        if self.epsilon == 0.0:
            # Evaluation mode: greedy
            action   = int(probs.argmax())
            log_prob = float(torch.log(probs[action] + 1e-8))
        else:
            # Training mode: sample from the distribution
            dist     = torch.distributions.Categorical(probs)
            action   = int(dist.sample())
            log_prob = float(dist.log_prob(torch.tensor(action)))

        self._pending.append((obs.copy(), action, log_prob, phase))
        return action

    # ------------------------------------------------------------------
    # Feedback hooks
    # ------------------------------------------------------------------

    def on_trick_result(self, won_trick: bool, trick_reward: float = 0.0):
        pass   # PPO uses round-level rewards; trick rewards ignored here

    def on_round_end(self, reward: float):
        """Assign round reward to all transitions collected this round."""
        for obs, action, log_prob, phase in self._pending:
            self._trajectory.append((obs, action, log_prob, reward, phase))
        self._pending = []

    def on_episode_end(self):
        if not self._trajectory:
            return

        obs_arr  = torch.FloatTensor(np.array([t[0] for t in self._trajectory]))
        actions  = torch.LongTensor( [t[1] for t in self._trajectory])
        old_lps  = torch.FloatTensor([t[2] for t in self._trajectory])
        rewards  = torch.FloatTensor([t[3] for t in self._trajectory])
        phases   = torch.FloatTensor([t[4] for t in self._trajectory])

        # --- Compute advantages (reward - baseline) ---
        with torch.no_grad():
            values = self.value_net(obs_arr)
        advantages = rewards - values
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- PPO policy update (bidding) ---
        bid_mask = phases < 0.5
        if bid_mask.any():
            self._ppo_update(self.bid_policy, self.bid_opt,
                             obs_arr[bid_mask], actions[bid_mask],
                             old_lps[bid_mask], advantages[bid_mask])

        # --- PPO policy update (playing) ---
        play_mask = phases >= 0.5
        if play_mask.any():
            self._ppo_update(self.play_policy, self.play_opt,
                             obs_arr[play_mask], actions[play_mask],
                             old_lps[play_mask], advantages[play_mask])

        # --- Value network update ---
        pred  = self.value_net(obs_arr)
        vloss = nn.MSELoss()(pred, rewards)
        self.value_opt.zero_grad()
        vloss.backward()
        self.value_opt.step()

        self._trajectory = []

    # ------------------------------------------------------------------
    # PPO clipped objective
    # ------------------------------------------------------------------

    def _ppo_update(self, policy, optimizer, obs, actions, old_log_probs, advantages):
        """One gradient step of the clipped PPO surrogate objective."""
        logits   = policy(obs)
        probs    = torch.softmax(logits, dim=-1)
        dist     = torch.distributions.Categorical(probs)
        new_lps  = dist.log_prob(actions)

        # Probability ratio r(θ) = π_new(a|s) / π_old(a|s)
        ratio  = torch.exp(new_lps - old_log_probs)

        # Clipped surrogate loss
        surr1  = ratio * advantages
        surr2  = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss   = -torch.min(surr1, surr2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            'bid_policy':  self.bid_policy.state_dict(),
            'play_policy': self.play_policy.state_dict(),
            'value_net':   self.value_net.state_dict(),
        }, path)
        print(f"[PPOAgent] saved to {path}")

    def load(self, path: str):
        data = torch.load(path, weights_only=True)
        self.bid_policy.load_state_dict(data['bid_policy'])
        self.play_policy.load_state_dict(data['play_policy'])
        self.value_net.load_state_dict(data['value_net'])
        print(f"[PPOAgent] loaded from {path}")
