"""
PPO Agent  —  Hybrid: 186-feature exact observation + 5-type action space.

Observation spaces:
  PLAY_OBS_SIZE = 186  (game state 6 + hand 60 + trick 60 + burned 60)
  BID_OBS_SIZE  =  66  (game state 6 + hand 60 — trick/burned irrelevant for bidding)

Action space (play phase): 5 card types  [wizard, jester, trump, high, low]
  The engine maps the chosen type to a concrete card automatically.
  This keeps the action space simple while the rich obs gives the policy
  full information about exactly which cards are in hand / on the table.

Tactical priors (play phase only):
  wizard : always wins — boost when tricks needed, dump otherwise
  jester : always loses — dump when tricks needed, keep otherwise
  trump  : strong card — moderate boost when winning, penalty when dumping
  high   : above average — small boost when winning, small penalty when dumping
  low    : weak card — penalty when winning, small boost when dumping

Other features:
  - GAE (lam=0.95) for low-variance advantage estimates
  - Multi-episode buffer (buffer_episodes=8) for stable batches
  - Entropy decay tied to train_episodes so it hits the floor at end of training
  - Separate value networks per phase
  - k_epochs=8 to fully exploit each collected batch
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from common.base_agent import BaseAgent, MAX_BID, NUM_CARD_TYPES, CARD_TYPES

PLAY_OBS_SIZE = 186   # full exact obs
BID_OBS_SIZE  = 66    # game state (6) + hand (60)


# ------------------------------------------------------------------
# Networks
# ------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


class ValueNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------

class PPOAgent(BaseAgent):

    def __init__(self, name,
                 lr=3e-4,
                 gamma=0.98,
                 clip_eps=0.2,
                 hidden_size=256,
                 k_epochs=8,
                 entropy_coef=0.05,
                 entropy_min=0.001,
                 train_episodes=10000,
                 tactic_scale=1.0,
                 lam=0.95,
                 buffer_episodes=8):
        super().__init__(name)

        self.uses_exact_obs = True   # hybrid: 186-feature obs, 5-type actions

        self.gamma           = gamma
        self.clip_eps        = clip_eps
        self.k_epochs        = k_epochs
        self.entropy_coef    = entropy_coef
        self.entropy_min     = entropy_min
        self.entropy_decay   = (entropy_min / entropy_coef) ** (1.0 / train_episodes)
        self.tactic_scale    = tactic_scale
        self.epsilon         = 1.0
        self.lam             = lam
        self.buffer_episodes = buffer_episodes

        # Bid network: trimmed obs (game state + hand only)
        # Play network: full 186-feature obs, 5-type output
        self.bid_policy      = PolicyNetwork(BID_OBS_SIZE,   MAX_BID + 1,     hidden_size)
        self.play_policy     = PolicyNetwork(PLAY_OBS_SIZE,  NUM_CARD_TYPES,  hidden_size)
        self.bid_value_net   = ValueNetwork(BID_OBS_SIZE,                     hidden_size)
        self.play_value_net  = ValueNetwork(PLAY_OBS_SIZE,                    hidden_size)

        self.bid_opt       = optim.Adam(self.bid_policy.parameters(),     lr=lr)
        self.play_opt      = optim.Adam(self.play_policy.parameters(),    lr=lr)
        self.bid_val_opt   = optim.Adam(self.bid_value_net.parameters(),  lr=lr)
        self.play_val_opt  = optim.Adam(self.play_value_net.parameters(), lr=lr)

        self._round_buffer       = []
        self._pending_trick      = None
        self._trajectory         = []
        self._all_steps          = []
        self._episodes_collected = 0

    # ------------------------------------------------------------------
    # Tactical prior (play phase only)
    # ------------------------------------------------------------------

    def _play_tactical_bonus(self, obs: np.ndarray,
                             valid_types: list) -> torch.Tensor:
        """
        Per-type logit adjustment based on tricks_needed.
        obs[4] = tricks_needed (normalised by /2, range [-1, 1])
        """
        tricks_needed = obs[4] * 2.0   # un-normalise back to [-2, 2]
        s = self.tactic_scale
        bonus = torch.zeros(NUM_CARD_TYPES)

        for type_idx in valid_types:
            name = CARD_TYPES[type_idx]
            if name == 'wizard':   # always wins
                bonus[type_idx] = 3.0 * s if tricks_needed > 0 else -3.0 * s
            elif name == 'jester': # always loses
                bonus[type_idx] = -2.0 * s if tricks_needed > 0 else 3.0 * s
            elif name == 'trump':  # strong
                bonus[type_idx] = 2.0 * s if tricks_needed > 0 else -2.0 * s
            elif name == 'high':   # above average
                bonus[type_idx] = 1.0 * s if tricks_needed > 0 else -1.0 * s
            else:                  # low
                bonus[type_idx] = -1.0 * s if tricks_needed > 0 else 1.0 * s

        return bonus

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray, valid_actions) -> int:
        phase = obs[1]   # index 1 in exact obs

        with torch.no_grad():
            if phase < 0.5:
                # BIDDING — use trimmed obs (game state + hand)
                bid_obs = torch.FloatTensor(obs[:BID_OBS_SIZE]).unsqueeze(0)
                logits  = self.bid_policy(bid_obs)

                mask = torch.full_like(logits, float('-inf'))
                for a in valid_actions:
                    mask[0, a] = 0.0
                logits = logits + mask

                dist     = torch.distributions.Categorical(logits=logits)
                action   = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action)).item()

            else:
                # PLAYING — full 186-feature obs + 5-type actions + tactical prior
                play_obs = torch.FloatTensor(obs).unsqueeze(0)
                logits   = self.play_policy(play_obs)

                mask = torch.full((1, NUM_CARD_TYPES), float('-inf'))
                for a in valid_actions:
                    mask[0, a] = 0.0
                logits = logits + mask
                logits = logits + self._play_tactical_bonus(obs, valid_actions).unsqueeze(0)

                dist     = torch.distributions.Categorical(logits=logits)
                action   = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action)).item()

        if self.epsilon != 0.0:
            if phase < 0.5:
                if self._pending_trick is not None:
                    p_obs, p_act, p_lp, p_phase = self._pending_trick
                    self._round_buffer.append({
                        'obs': p_obs, 'action': p_act,
                        'log_prob': p_lp, 'reward': 0.0, 'phase': p_phase,
                    })
                    self._pending_trick = None
                self._round_buffer.append({
                    'obs': obs, 'action': action,
                    'log_prob': log_prob, 'reward': 0.0, 'phase': phase,
                })
            else:
                self._pending_trick = (obs, action, log_prob, phase)

        return action

    # ------------------------------------------------------------------
    # Feedback hooks
    # ------------------------------------------------------------------

    def on_trick_result(self, won_trick: bool, trick_reward: float = 0.0,
                        trick_card_types: list = None):
        if self._pending_trick is None:
            return
        p_obs, p_act, p_lp, p_phase = self._pending_trick
        self._round_buffer.append({
            'obs': p_obs, 'action': p_act,
            'log_prob': p_lp, 'reward': trick_reward, 'phase': p_phase,
        })
        self._pending_trick = None

    def on_round_end(self, reward: float):
        if not self._round_buffer:
            return
        if self._pending_trick is not None:
            p_obs, p_act, p_lp, p_phase = self._pending_trick
            self._round_buffer.append({
                'obs': p_obs, 'action': p_act,
                'log_prob': p_lp, 'reward': 0.0, 'phase': p_phase,
            })
            self._pending_trick = None
        self._round_buffer[-1]['reward'] += reward
        for entry in self._round_buffer:
            self._trajectory.append(entry)
        self._round_buffer = []

    def on_episode_end(self):
        trajectory           = self._trajectory
        self._trajectory     = []
        self._round_buffer   = []
        self._pending_trick  = None

        self.entropy_coef = max(self.entropy_min,
                                self.entropy_coef * self.entropy_decay)

        if not trajectory:
            return

        trajectory[-1]['done'] = True
        self._all_steps.extend(trajectory)
        self._episodes_collected += 1

        if self._episodes_collected < self.buffer_episodes:
            return

        self._episodes_collected = 0
        all_steps       = self._all_steps
        self._all_steps = []
        self._run_update(all_steps)

    # ------------------------------------------------------------------
    # GAE + PPO update
    # ------------------------------------------------------------------

    def _run_update(self, steps: list):
        T = len(steps)

        obs_arr  = torch.FloatTensor(np.array([s['obs']       for s in steps]))
        actions  = torch.LongTensor(           [s['action']   for s in steps])
        old_lps  = torch.FloatTensor(          [s['log_prob'] for s in steps])
        rewards  = np.array(                   [s['reward']   for s in steps], dtype=np.float32)
        phases   = torch.FloatTensor(          [s['phase']    for s in steps])
        done     = np.array(                   [s.get('done', False) for s in steps], dtype=bool)

        bid_mask  = phases < 0.5
        play_mask = phases >= 0.5

        # Bid network sees trimmed obs; play network sees full obs
        bid_obs  = obs_arr[:, :BID_OBS_SIZE]
        play_obs = obs_arr

        with torch.no_grad():
            values_t = torch.zeros(T)
            if bid_mask.any():
                values_t[bid_mask]  = self.bid_value_net(bid_obs[bid_mask])
            if play_mask.any():
                values_t[play_mask] = self.play_value_net(play_obs[play_mask])

        values_np = values_t.numpy()

        # GAE with episode boundary handling
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            is_terminal = bool(done[t])
            next_val    = 0.0 if is_terminal else (values_np[t + 1] if t + 1 < T else 0.0)
            carry       = 0.0 if is_terminal else gae
            delta       = rewards[t] + self.gamma * next_val - values_np[t]
            gae         = delta + self.gamma * self.lam * carry
            advantages[t] = gae

        returns_arr = torch.FloatTensor(advantages + values_np)
        adv_t       = torch.FloatTensor(advantages)
        if adv_t.std() > 1e-8:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(self.k_epochs):
            if bid_mask.any():
                self._ppo_update(self.bid_policy, self.bid_opt,
                                 bid_obs[bid_mask], actions[bid_mask],
                                 old_lps[bid_mask], adv_t[bid_mask])

            if play_mask.any():
                self._ppo_update(self.play_policy, self.play_opt,
                                 play_obs[play_mask], actions[play_mask],
                                 old_lps[play_mask], adv_t[play_mask])

            if bid_mask.any():
                pred  = self.bid_value_net(bid_obs[bid_mask])
                vloss = nn.MSELoss()(pred, returns_arr[bid_mask])
                self.bid_val_opt.zero_grad()
                vloss.backward()
                nn.utils.clip_grad_norm_(self.bid_value_net.parameters(), max_norm=0.5)
                self.bid_val_opt.step()

            if play_mask.any():
                pred  = self.play_value_net(play_obs[play_mask])
                vloss = nn.MSELoss()(pred, returns_arr[play_mask])
                self.play_val_opt.zero_grad()
                vloss.backward()
                nn.utils.clip_grad_norm_(self.play_value_net.parameters(), max_norm=0.5)
                self.play_val_opt.step()

    # ------------------------------------------------------------------
    # PPO clipped objective with entropy bonus
    # ------------------------------------------------------------------

    def _ppo_update(self, policy, optimizer, obs, actions, old_log_probs, advantages):
        logits       = policy(obs)
        dist         = torch.distributions.Categorical(logits=logits)
        new_lps      = dist.log_prob(actions)

        ratio        = torch.exp(new_lps - old_log_probs)
        surr1        = ratio * advantages
        surr2        = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss  = -torch.min(surr1, surr2).mean()
        entropy_loss = -dist.entropy().mean()
        loss         = policy_loss + self.entropy_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            'bid_policy':     self.bid_policy.state_dict(),
            'play_policy':    self.play_policy.state_dict(),
            'bid_value_net':  self.bid_value_net.state_dict(),
            'play_value_net': self.play_value_net.state_dict(),
            'epsilon':        self.epsilon,
            'entropy_coef':   self.entropy_coef,
        }, path)
        print(f"[PPOAgent] saved to {path}")

    def load(self, path: str):
        data = torch.load(path, weights_only=True)
        self.bid_policy.load_state_dict(data['bid_policy'])
        self.play_policy.load_state_dict(data['play_policy'])
        self.bid_value_net.load_state_dict(data['bid_value_net'])
        self.play_value_net.load_state_dict(data['play_value_net'])
        self.epsilon      = data.get('epsilon',      1.0)
        self.entropy_coef = data.get('entropy_coef', self.entropy_coef)
        print(f"[PPOAgent] loaded from {path}")