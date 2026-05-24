"""
HeuristicAgent — rule-based Wizard player.

Uses hand-quality estimates and trick-need logic to make decisions.
Stronger than random but fully deterministic, so a trained PPO can beat it.

Bidding logic:
  expected_tricks = wizards * 1.0  +  trump_cards * 0.6  +  high_cards * 0.25
  Bid the rounded expected value, clamped to valid range.

Playing logic:
  Need more tricks  → prefer wizard > trump > high > low > jester
  At / over bid     → prefer jester > low > high > trump > wizard
"""

from common.base_agent import (BaseAgent, OBS_PHASE, OBS_TRICKS_NEEDED,
                                OBS_HAND_WIZARDS, OBS_HAND_HIGH, OBS_HAND_TRUMP,
                                OBS_TRICK_WIZARD)

# CARD_TYPES indices: wizard=0  jester=1  trump=2  high=3  low=4
_WIN_PRIORITY  = [0, 2, 3, 4, 1]   # strongest first
_DUMP_PRIORITY = [1, 4, 3, 2, 0]   # weakest first


class HeuristicAgent(BaseAgent):

    def act(self, obs, valid_actions):
        if obs[OBS_PHASE] < 0.5:
            return self._bid(obs, valid_actions)
        return self._play(obs, valid_actions)

    def _bid(self, obs, valid_actions):
        wizards = obs[OBS_HAND_WIZARDS] * 4.0
        trump   = obs[OBS_HAND_TRUMP]   * 10.0
        high    = obs[OBS_HAND_HIGH]    * 10.0
        expected = wizards * 1.0 + trump * 0.6 + high * 0.25
        bid = int(round(expected))
        bid = max(valid_actions[0], min(valid_actions[-1], bid))
        return bid

    def _play(self, obs, valid_actions):
        tricks_needed  = obs[OBS_TRICKS_NEEDED] * 2.0   # un-normalise from /2
        wizard_in_trick = obs[OBS_TRICK_WIZARD] > 0.5

        # Trick already won by a wizard — always dump weakest card
        if wizard_in_trick:
            priority = _DUMP_PRIORITY
        elif tricks_needed > 0:
            priority = _WIN_PRIORITY
        else:
            priority = _DUMP_PRIORITY

        for card_type in priority:
            if card_type in valid_actions:
                return card_type
        return valid_actions[0]
