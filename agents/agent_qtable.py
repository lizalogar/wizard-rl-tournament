"""
Q-Table Agent  —  YOUR FILE TO IMPLEMENT.

The engine always calls act(obs, valid_actions) and the three on_* hooks.
Do NOT touch any file in common/.

Quick-start guide
-----------------
obs is a numpy float32 array of length 12. Use the OBS_* constants to
read individual features, e.g.:

    round_num = round(obs[OBS_ROUND_NUM] * 10)   # integer 1..10
    phase     = obs[OBS_PHASE]                    # 0.0=bid, 1.0=play

During bidding (phase == 0): valid_actions = [0, 1, ..., round_num]
During playing (phase == 1): valid_actions = subset of [0,1,2,3,4]
                              (indices into CARD_TYPES = ['wizard','jester','trump','high','low'])

Your act() must return one integer from valid_actions.

Suggested improvements over the starter code below:
  - Tune alpha, gamma, epsilon_decay with grid_search() in tournament.py
  - Extend the bid or play state with more obs features
  - Try different reward shaping in on_trick_result / on_round_end
"""

import random
import pickle
import numpy as np
from common.base_agent import (BaseAgent,
                                OBS_ROUND_NUM, OBS_PHASE, OBS_HAND_WIZARDS,
                                OBS_HAND_HIGH, OBS_HAND_TRUMP,
                                OBS_TRICKS_NEEDED, OBS_POSITION,
                                OBS_HAVE_WIZARD, OBS_HAVE_JESTER, OBS_HAVE_TRUMP)


class QTableAgent(BaseAgent):
    """
    Q-learning agent with two separate Q-tables:
      bid_q   — maps bid-state tuple  to {bid_amount: Q-value}
      play_q  — maps play-state tuple to {card_type_index: Q-value}

    Both tables start empty (all unseen states implicitly have Q = 0).
    The agent uses epsilon-greedy exploration: with probability epsilon it
    picks randomly, otherwise it picks the action with the highest Q-value.
    Epsilon decays after every game so the agent explores less as it learns.
    """

    def __init__(self, name,
                 alpha=0.05,           # learning rate
                 gamma=0.9,           # discount factor 0.9
                 epsilon=0.9,         # starting exploration rate
                 epsilon_decay=0.999,
                 epsilon_min=0.01):
        super().__init__(name)
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min

        self.bid_q  = {}   # {state_tuple: {action_int: float}}
        self.play_q = {}

        # Filled in act() and consumed in on_round_end()
        self._bid_experience   = None   # (state, action)
        self._play_experiences = []     # [(state, action), ...]
        self._round_reward     = 0.0

    # ------------------------------------------------------------------
    # State builders  (convert obs back to discrete keys for the tables)
    # ------------------------------------------------------------------

    def _bid_state(self, obs) -> tuple:
        """
        Improved bid state: Expanded the resolution for high cards and trump cards 
        to allow the agent to better distinguish between 'good' and 'great' hands 
        in the later rounds when hands are larger.
        """
        round_num   = round(obs[OBS_ROUND_NUM] * 10)
        num_wizards = round(obs[OBS_HAND_WIZARDS] * 4)
        num_high    = round(obs[OBS_HAND_HIGH] * 10)
        num_trump   = round(obs[OBS_HAND_TRUMP] * 10)
        
        # We clamp these to smaller bins (0, 1, 2, 3, 4, 5+) to keep the table manageable
        return (round_num,
                min(num_wizards, 2),
                min(num_high,    5),
                min(num_trump,   5))

    def _play_state(self, obs) -> tuple:
        """
        Improved play state: Added 'is_late_game' context. Playing a card when 
        you only have 2 cards total (early game) requires different logic than 
        when you are managing a 10-card hand (late game)
        """
        round_num     = round(obs[OBS_ROUND_NUM] * 10)
        is_late_game  = bool(round_num > 5) # True for rounds 6-10

        tricks_needed = round(obs[OBS_TRICKS_NEEDED] * 2)   # -2..2
        position      = round(obs[OBS_POSITION]      * 2)   # 0..2
        
        have_wizard_valid = bool(obs[OBS_HAVE_WIZARD] > 0.5)
        have_jester_valid = bool(obs[OBS_HAVE_JESTER] > 0.5)
        have_trump_valid  = bool(obs[OBS_HAVE_TRUMP]  > 0.5)
        
        #Broad categorization of remaining hand strength
        has_high_in_hand  = bool(obs[OBS_HAND_HIGH] > 0.0)
        has_trump_in_hand = bool(obs[OBS_HAND_TRUMP] > 0.0)

        return (is_late_game,tricks_needed, position, 
                have_wizard_valid, have_jester_valid, have_trump_valid,
                has_high_in_hand, has_trump_in_hand)

    # ------------------------------------------------------------------
    # Q-table helpers
    # ------------------------------------------------------------------

    def _q(self, table, state, action) -> float:
        return table.get(state, {}).get(action, 0.0)

    def _best_action(self, table, state, actions) -> int:
        return max(actions, key=lambda a: self._q(table, state, a))

    def _update_q(self, table, state, action, reward):
        if state not in table:
            table[state] = {}
        old = table[state].get(action, 0.0)
        table[state][action] = old + self.alpha * (reward - old)

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray, valid_actions: list) -> int:
        phase = obs[OBS_PHASE]

        if phase < 0.5:
            # ---- Bidding ----
            state = self._bid_state(obs)
            if random.random() < self.epsilon:
                action = random.choice(valid_actions)
            else:
                action = self._best_action(self.bid_q, state, valid_actions)
            self._bid_experience   = (state, action)
            self._play_experiences = []
        else:
            # ---- Playing ----
            state = self._play_state(obs)
            if random.random() < self.epsilon:
                action = random.choice(valid_actions)
            else:
                action = self._best_action(self.play_q, state, valid_actions)
            self._play_experiences.append((state, action))

        return action

    def on_trick_result(self, won_trick: bool, trick_reward: float = 0.0):
        if not self._play_experiences:
            return
        state, action = self._play_experiences[-1]
        # Reward shaping: apply a small multiplier to trick rewards to balance 
        # against the massive round-end rewards (which range from -100 to +120)
        shaped_trick_reward = trick_reward * 0.5 
        self._update_q(self.play_q, state, action, shaped_trick_reward)

    def on_round_end(self, reward: float):
        """
        Discounted credit assignment: last play gets full reward, earlier
        plays get exponentially less. Bid gets the raw round reward.
        """
        if self._bid_experience:
            state, action = self._bid_experience
            self._update_q(self.bid_q, state, action, reward)

        n = len(self._play_experiences)
        for i, (state, action) in enumerate(self._play_experiences):
            steps_from_end = n - 1 - i
            self._update_q(self.play_q, state, action,
                           reward * (self.gamma ** steps_from_end))


    def on_episode_end(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'bid_q': self.bid_q,
                         'play_q': self.play_q,
                         'epsilon': self.epsilon}, f)
        print(f"[QTableAgent] saved to {path}")

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.bid_q   = data['bid_q']
        self.play_q  = data['play_q']
        self.epsilon = data['epsilon']
        print(f"[QTableAgent] loaded from {path} (epsilon={self.epsilon:.3f})")
