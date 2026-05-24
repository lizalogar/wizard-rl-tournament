"""
BaseAgent — the shared interface every agent must implement.

DO NOT MODIFY THIS FILE.  It is shared code that all agents depend on.
Your agent lives in agents/agent_<yourname>.py and inherits from this class.
"""

from abc import ABC, abstractmethod
import numpy as np

# ------------------------------------------------------------------
# Observation vector  (15 floats, all roughly in [0, 1])
# ------------------------------------------------------------------
# Index constants so agents don't have to use magic numbers.

OBS_ROUND_NUM     = 0   # round number / 10          (0.1 … 1.0)
OBS_PHASE         = 1   # 0.0 = bidding, 1.0 = playing
OBS_MY_BID        = 2   # current bid / 10
OBS_TRICKS_WON    = 3   # tricks won so far / 10
OBS_TRICKS_NEEDED = 4   # (bid - tricks_won) clamped [-2, 2] / 2
OBS_POSITION      = 5   # position in current trick  (0, 1, 2) / 2
OBS_HAND_WIZARDS  = 6   # wizard count in hand / 4
OBS_HAND_HIGH     = 7   # high cards (value 10-13) in hand / 10
OBS_HAND_TRUMP    = 8   # trump-suit cards in hand / 10
OBS_HAVE_WIZARD   = 9   # 1.0 if a wizard is in valid_cards
OBS_HAVE_JESTER   = 10  # 1.0 if a jester is in valid_cards
OBS_HAVE_TRUMP    = 11  # 1.0 if a trump card is in valid_cards
# --- NEW FEATURES ---
OBS_TRICK_HAS_LEAD        = 12  # 1.0 if a lead suit has been established
OBS_WINNING_CARD_VALUE    = 13  # Value of current winning card / 14
OBS_WINNING_CARD_IS_TRUMP = 14  # 1.0 if the current winning card is a trump card
OBS_SIZE                  = 15  # Increased from 12

# ------------------------------------------------------------------
# Action encoding
# ------------------------------------------------------------------
# BIDDING phase  (obs[OBS_PHASE] == 0.0):
#   valid_actions = [0, 1, ..., round_number]
#   act() must return one of those integers (your bid)
#
# PLAYING phase  (obs[OBS_PHASE] == 1.0):
#   valid_actions = subset of {0, 1, 2, 3, 4}  (card-type indices)
#   act() must return one of those integers
#   The engine maps the index to an actual card automatically.

CARD_TYPES     = ['wizard', 'jester', 'trump', 'high', 'low']
NUM_CARD_TYPES = 5    # size of the play action space
MAX_BID        = 11   # size of the bid action space (bids 0..10)


# ------------------------------------------------------------------
# Abstract base class
# ------------------------------------------------------------------

class BaseAgent(ABC):
    """
    Every agent must subclass this and implement act().

    The engine calls:
      act()            — to get a bid or card-type decision
      on_trick_result()  — immediately after each trick resolves
      on_round_end()   — after all tricks in a round, with score delta
      on_episode_end() — after all 10 rounds in a game
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def act(self, obs: np.ndarray, valid_actions: list) -> int:
        """
        Return one action from valid_actions.

        obs           float32 array of length OBS_SIZE (see constants above)
        valid_actions list of ints — only these indices are legal right now

        During BIDDING  (obs[OBS_PHASE] == 0.0)
            valid_actions = [0, 1, ..., round_number]
            return        = your bid (integer)

        During PLAYING  (obs[OBS_PHASE] == 1.0)
            valid_actions = subset of [0, 1, 2, 3, 4]  (CARD_TYPES indices)
            return        = chosen card-type index (integer)
            The engine picks a random card of that type from your hand.
        """

    def on_trick_result(self, won_trick: bool, trick_reward: float = 0.0):
        """
        Called right after each trick resolves.

        won_trick    True if this agent won the trick
        trick_reward immediate reward already computed (+15 / -15 / +10 / -5)
                     based on whether winning this trick helps your bid
        """

    def on_round_end(self, reward: float):
        """
        Called after scoring with this agent's score change for the round.

        reward > 0  bid was exactly right  (+20 + 10*tricks_won)
        reward < 0  bid was wrong          (-10 per trick off)
        """

    def on_episode_end(self):
        """Called once after all 10 rounds. Good place to decay epsilon."""

    def save(self, path: str):
        """Persist weights / Q-tables to path."""

    def load(self, path: str):
        """Restore weights / Q-tables from path."""
