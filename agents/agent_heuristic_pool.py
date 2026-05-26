"""
Heuristic agent variants for diverse opponent training.

All share the same card-play framework as HeuristicAgent but differ in
bidding aggression or card-type preferences, giving the PPO agent a wider
range of playstyles to adapt to during phase 2 curriculum training.

Card-type indices:  wizard=0  jester=1  trump=2  high=3  low=4
"""

from common.base_agent import (BaseAgent, OBS_PHASE, OBS_TRICKS_NEEDED,
                                OBS_HAND_WIZARDS, OBS_HAND_HIGH, OBS_HAND_TRUMP,
                                OBS_TRICK_WIZARD, OBS_ROUND_NUM)

_WIN_PRIORITY   = [0, 2, 3, 4, 1]   # wizard > trump > high > low > jester
_DUMP_PRIORITY  = [1, 4, 3, 2, 0]   # jester > low > high > trump > wizard
_TRUMP_PRIORITY = [2, 0, 3, 4, 1]   # trump > wizard > high > low > jester


def _expected_tricks(obs):
    wizards = obs[OBS_HAND_WIZARDS] * 4.0
    trump   = obs[OBS_HAND_TRUMP]   * 10.0
    high    = obs[OBS_HAND_HIGH]    * 10.0
    return wizards * 1.0 + trump * 0.6 + high * 0.25


def _play_standard(obs, valid_actions):
    """Shared play logic: win when needed, dump otherwise."""
    tricks_needed   = obs[OBS_TRICKS_NEEDED] * 2.0
    wizard_in_trick = obs[OBS_TRICK_WIZARD] > 0.5
    if wizard_in_trick:
        priority = _DUMP_PRIORITY
    elif tricks_needed > 0:
        priority = _WIN_PRIORITY
    else:
        priority = _DUMP_PRIORITY
    for ct in priority:
        if ct in valid_actions:
            return ct
    return valid_actions[0]


class AggressiveBidderAgent(BaseAgent):
    """Always bids one higher than expected — constantly over-reaches."""

    def act(self, obs, valid_actions):
        if obs[OBS_PHASE] < 0.5:
            expected = _expected_tricks(obs)
            bid = min(valid_actions[-1], int(round(expected)) + 1)
            return max(valid_actions[0], bid)
        return _play_standard(obs, valid_actions)


class ConservativeBidderAgent(BaseAgent):
    """Always bids one lower than expected — plays safe, sheds tricks."""

    def act(self, obs, valid_actions):
        if obs[OBS_PHASE] < 0.5:
            expected = _expected_tricks(obs)
            bid = max(valid_actions[0], int(round(expected)) - 1)
            return min(valid_actions[-1], bid)
        return _play_standard(obs, valid_actions)


class TrumpHeavyAgent(BaseAgent):
    """
    Bids like the standard heuristic but plays trump cards aggressively —
    leads with trump even when not strictly needed, as long as tricks are
    still available to win.
    """

    def act(self, obs, valid_actions):
        if obs[OBS_PHASE] < 0.5:
            expected = _expected_tricks(obs)
            bid = int(round(expected))
            return max(valid_actions[0], min(valid_actions[-1], bid))

        tricks_needed   = obs[OBS_TRICKS_NEEDED] * 2.0
        wizard_in_trick = obs[OBS_TRICK_WIZARD] > 0.5

        if wizard_in_trick:
            priority = _DUMP_PRIORITY
        elif tricks_needed > 0:
            priority = _TRUMP_PRIORITY   # leads with trump instead of wizard
        else:
            priority = _DUMP_PRIORITY
        for ct in priority:
            if ct in valid_actions:
                return ct
        return valid_actions[0]
