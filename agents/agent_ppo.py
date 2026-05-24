"""
PPO Agent  —  Enhanced implementation.

Improvements over the skeleton:
  - 3-layer 128-unit networks with LayerNorm for stable gradients
  - Trick rewards used via on_trick_result(); discounted backward within each
    round so early steps get proper credit, not a flat round reward
  - K=4 update epochs per episode
  - Entropy bonus to prevent premature policy collapse
  - Gradient clipping (max_norm=0.5)
  - Tactical logit priors: bias toward winning/shedding cards based on
    how many tricks are still needed, and toward sensible bids based on
    hand quality
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from common.base_agent import (BaseAgent, OBS_SIZE, OBS_PHASE,
                                OBS_TRICKS_NEEDED,
                                OBS_HAND_WIZARDS, OBS_HAND_HIGH, OBS_HAND_TRUMP,
                                NUM_CARD_TYPES, MAX_BID)


# ------------------------------------------------------------------
# Networks
# ------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
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
    def __init__(self, input_size, hidden_size=128):
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
    """
    PPO agent with:
      - separate bid / play policy networks
      - shared value network
      - per-trick reward tracking with within-round discounted returns
      - K-epoch updates, entropy bonus, gradient clipping
      - tactical logit priors

    self.epsilon: 1.0 = stochastic training, 0.0 = greedy evaluation
    """

    def __init__(self, name,
                 lr=3e-4,
                 gamma=0.95,
                 clip_eps=0.2,
                 hidden_size=128,
                 k_epochs=4,
                 entropy_coef=0.01,
                 tactic_scale=0.5,
                 obs_augment=True):
        super().__init__(name)
        self.gamma        = gamma
        self.clip_eps     = clip_eps
        self.k_epochs     = k_epochs
        self.entropy_coef = entropy_coef
        self.tactic_scale = tactic_scale
        self.epsilon      = 1.0   # tournament sets this to 0 for evaluation
        self.obs_augment  = obs_augment

        _in = OBS_SIZE + 2 if obs_augment else OBS_SIZE
        self.bid_policy  = PolicyNetwork(_in, MAX_BID,        hidden_size)
        self.play_policy = PolicyNetwork(_in, NUM_CARD_TYPES, hidden_size)
        self.value_net   = ValueNetwork(_in,                  hidden_size)

        self.bid_opt   = optim.Adam(self.bid_policy.parameters(),  lr=lr)
        self.play_opt  = optim.Adam(self.play_policy.parameters(), lr=lr)
        self.value_opt = optim.Adam(self.value_net.parameters(),   lr=lr)

        # Round-level card counts tracked by the agent from trick_card_types callbacks
        self._round_wizards_seen = 0   # wizards played by ALL players this round so far
        self._round_trump_seen   = 0   # trump cards played by ALL players this round so far

        # Within-round buffer: list of dicts
        # {'obs', 'action', 'log_prob', 'trick_reward', 'phase'}
        self._round_buffer  = []

        # Holds the most recent play-step until on_trick_result delivers its reward
        # (obs, action, log_prob, phase)
        self._pending_trick = None

        # Episode-level finalized trajectory
        # {'obs', 'action', 'log_prob', 'return_', 'phase'}
        self._trajectory = []

    # ------------------------------------------------------------------
    # Tactical logit bonus
    # ------------------------------------------------------------------

    def _tactical_bonus(self, obs: np.ndarray, phase: float) -> torch.Tensor:
        """Soft logit adjustments that encode Wizard strategy."""
        if phase < 0.5:
            # Bidding: penalise bids far from expected trick count
            wizards  = float(obs[OBS_HAND_WIZARDS]) * 4.0
            trump    = float(obs[OBS_HAND_TRUMP])   * 10.0
            high     = float(obs[OBS_HAND_HIGH])    * 10.0
            expected = wizards * 1.0 + trump * 0.6 + high * 0.3

            bonus = torch.zeros(MAX_BID)
            for b in range(MAX_BID):
                bonus[b] = float(-abs(b - expected) * self.tactic_scale)
            return bonus

        # Playing phase: steer toward winning or shedding based on need
        # [0=wizard, 1=jester, 2=trump, 3=high, 4=low]
        tricks_needed = obs[OBS_TRICKS_NEEDED] * 2.0   # un-normalise from /2

        bonus = torch.zeros(NUM_CARD_TYPES)
        s = self.tactic_scale
        if tricks_needed > 0:
            bonus[0] =  3.0 * s   # wizard
            bonus[2] =  2.0 * s   # trump
            bonus[3] =  1.0 * s   # high
            bonus[1] = -2.0 * s   # jester
            bonus[4] = -1.0 * s   # low
        else:
            bonus[1] =  3.0 * s   # jester
            bonus[4] =  1.0 * s   # low
            bonus[0] = -3.0 * s   # wizard
            bonus[2] = -2.0 * s   # trump
            bonus[3] = -1.0 * s   # high
        return bonus

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        if not self.obs_augment:
            return obs
        extra = np.array([
            self._round_wizards_seen / 4.0,
            self._round_trump_seen   / 13.0,
        ], dtype=np.float32)
        return np.concatenate([obs, extra])

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray, valid_actions: list) -> int:
        phase  = obs[OBS_PHASE]

        # Reset round counters at the start of each new round (bid phase)
        if phase < 0.5:
            self._round_wizards_seen = 0
            self._round_trump_seen   = 0

        aug_obs = self._augment_obs(obs)
        obs_t   = torch.FloatTensor(aug_obs)
        policy = self.bid_policy if phase < 0.5 else self.play_policy

        with torch.no_grad():
            logits = policy(obs_t)

        # Tactical prior (uses original obs indices — unaffected by augmentation)
        logits = logits + self._tactical_bonus(obs, phase)

        # Mask invalid actions
        mask = torch.full((logits.shape[0],), float('-inf'))
        for a in valid_actions:
            mask[a] = 0.0
        logits = logits + mask

        probs = torch.softmax(logits, dim=-1)

        if self.epsilon == 0.0:
            action   = int(probs.argmax())
            log_prob = float(torch.log(probs[action] + 1e-8))
        else:
            dist     = torch.distributions.Categorical(probs)
            action   = int(dist.sample())
            log_prob = float(dist.log_prob(torch.tensor(action)))

        # Only collect trajectory during training.
        # When epsilon=0 (evaluation mode), act greedily but do NOT record
        # transitions — evaluate_vs_random calls on_episode_end() too, which
        # would otherwise trigger a surprise update against random opponents.
        if self.epsilon != 0.0:
            if phase < 0.5:
                if self._pending_trick is not None:
                    p_obs, p_act, p_lp, p_phase = self._pending_trick
                    self._round_buffer.append({
                        'obs': p_obs, 'action': p_act, 'log_prob': p_lp,
                        'trick_reward': 0.0, 'phase': p_phase,
                    })
                    self._pending_trick = None
                self._round_buffer.append({
                    'obs': aug_obs, 'action': action,
                    'log_prob': log_prob, 'trick_reward': 0.0, 'phase': phase,
                })
            else:
                self._pending_trick = (aug_obs, action, log_prob, phase)

        return action

    # ------------------------------------------------------------------
    # Feedback hooks
    # ------------------------------------------------------------------

    def on_trick_result(self, won_trick: bool, trick_reward: float = 0.0,
                        trick_card_types: list = None):
        types = list(trick_card_types) if trick_card_types else []

        # Update private round-level card counts (wizard=0, trump=2 per CARD_TYPES)
        self._round_wizards_seen += types.count(0)
        self._round_trump_seen   += types.count(2)

        if self._pending_trick is None:
            return
        p_obs, p_act, p_lp, p_phase = self._pending_trick
        self._round_buffer.append({
            'obs':              p_obs,
            'action':           p_act,
            'log_prob':         p_lp,
            'trick_reward':     trick_reward,
            'phase':            p_phase,
            'trick_card_types': types,
        })
        self._pending_trick = None

    def on_round_end(self, reward: float):
        """Compute within-round discounted returns and flush to trajectory."""
        if not self._round_buffer:
            return

        n = len(self._round_buffer)
        returns = [0.0] * n

        # Backward pass: round_reward is the terminal signal
        G = reward
        for t in range(n - 1, -1, -1):
            G = self._round_buffer[t]['trick_reward'] + self.gamma * G
            returns[t] = G

        for entry, G_t in zip(self._round_buffer, returns):
            self._trajectory.append({
                'obs':      entry['obs'],
                'action':   entry['action'],
                'log_prob': entry['log_prob'],
                'return_':  G_t,
                'phase':    entry['phase'],
            })

        self._round_buffer = []

    def on_episode_end(self):
        # Always clean up so stale data never bleeds into the next episode
        trajectory = self._trajectory
        self._trajectory         = []
        self._round_buffer       = []
        self._pending_trick      = None
        self._round_wizards_seen = 0
        self._round_trump_seen   = 0

        if not trajectory:
            return

        obs_arr = torch.FloatTensor(np.array([t['obs']      for t in trajectory]))
        actions = torch.LongTensor(            [t['action']  for t in trajectory])
        old_lps = torch.FloatTensor(           [t['log_prob'] for t in trajectory])
        returns = torch.FloatTensor(           [t['return_'] for t in trajectory])
        phases  = torch.FloatTensor(           [t['phase']   for t in trajectory])

        bid_mask  = phases < 0.5
        play_mask = phases >= 0.5

        for _ in range(self.k_epochs):
            # Recompute advantages each epoch (value net updates inside loop)
            with torch.no_grad():
                values = self.value_net(obs_arr)
            advantages = returns - values
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            if bid_mask.any():
                self._ppo_update(
                    self.bid_policy, self.bid_opt,
                    obs_arr[bid_mask], actions[bid_mask],
                    old_lps[bid_mask], advantages[bid_mask],
                )

            if play_mask.any():
                self._ppo_update(
                    self.play_policy, self.play_opt,
                    obs_arr[play_mask], actions[play_mask],
                    old_lps[play_mask], advantages[play_mask],
                )

            pred  = self.value_net(obs_arr)
            vloss = nn.MSELoss()(pred, returns)
            self.value_opt.zero_grad()
            vloss.backward()
            nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
            self.value_opt.step()

        self._trajectory = []

    # ------------------------------------------------------------------
    # PPO clipped objective with entropy bonus
    # ------------------------------------------------------------------

    def _ppo_update(self, policy, optimizer, obs, actions, old_log_probs, advantages):
        logits  = policy(obs)
        probs   = torch.softmax(logits, dim=-1)
        dist    = torch.distributions.Categorical(probs)
        new_lps = dist.log_prob(actions)

        ratio = torch.exp(new_lps - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss  = -torch.min(surr1, surr2).mean()
        entropy_loss = -dist.entropy().mean()   # we want to maximise entropy
        loss = policy_loss + self.entropy_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            'bid_policy':   self.bid_policy.state_dict(),
            'play_policy':  self.play_policy.state_dict(),
            'value_net':    self.value_net.state_dict(),
            'epsilon':      self.epsilon,
            'obs_augment':  self.obs_augment,
        }, path)
        print(f"[PPOAgent] saved to {path}")

    def load(self, path: str):
        data = torch.load(path, weights_only=True)
        self.bid_policy.load_state_dict(data['bid_policy'])
        self.play_policy.load_state_dict(data['play_policy'])
        self.value_net.load_state_dict(data['value_net'])
        self.epsilon     = data.get('epsilon',     1.0)
        self.obs_augment = data.get('obs_augment', False)
        print(f"[PPOAgent] loaded from {path}  (obs_augment={self.obs_augment})")
