"""
ClaudeAgent — LLM-strategy agent for the Wizard card game.

This agent encodes the reasoning Claude (the LLM) would apply when
playing Wizard: calibrated expected-value bidding and context-aware
card play with position/trick-state sensitivity.

It operates on the same 17-feature observation vector as the DQN/QTable
agents (no neural nets, no training required).

Key design choices
──────────────────
Bidding
  • Wizards are guaranteed wins (+1.0 each).
  • The best trump contributes proportionally to its rank (a 13-rank trump
    wins ~95 % of the time; a 1-rank trump wins ~40 %).
  • Additional trump cards each add ~0.35 expected tricks.
  • High non-trump cards (10–13) contribute ~0.25 each (they only win if
    no opponent plays trump or wizard on that trick).
  • A small conservative bias (−0.15 before rounding) is applied:
    under-bidding by 1 costs −10 pts; over-bidding by 1 also costs −10 pts,
    but over-bidding is slightly worse psychologically and compounds with
    later rounds, so erring slightly low is rational.

Playing
  ┌─ Trick already has a Wizard → can't beat it → dump weakest card.
  ├─ Need more tricks (tricks_needed > 0)
  │    • Leading      → play strongest available: Wizard > Trump > High > Low > Jester
  │    • Following, trick has Trump → only Wizard beats it; otherwise dump
  │    • Following, no Trump yet   → Wizard > Trump > High > Low > Jester
  └─ At or over bid (tricks_needed ≤ 0) → dump
       • Leading      → Jester > Low > High > Trump > Wizard
       • Following    → same dump order; trick already has trump? safe to dump Low
"""

import numpy as np
from common.base_agent import (
    BaseAgent,
    OBS_PHASE, OBS_TRICKS_NEEDED, OBS_POSITION,
    OBS_HAND_WIZARDS, OBS_HAND_HIGH, OBS_HAND_TRUMP,
    OBS_HAVE_WIZARD, OBS_HAVE_JESTER, OBS_HAVE_TRUMP,
    OBS_TRICK_WIZARD, OBS_TRICK_TRUMP,
    OBS_HAND_MAX_TRUMP,
)

# Card type indices (must match CARD_TYPES in base_agent.py)
WIZARD = 0
JESTER = 1
TRUMP  = 2
HIGH   = 3
LOW    = 4

WINNING_ORDER = [WIZARD, TRUMP, HIGH, LOW, JESTER]
DUMPING_ORDER = [JESTER, LOW, HIGH, TRUMP, WIZARD]


class ClaudeAgent(BaseAgent):
    """
    Rule-based strategic agent representing LLM reasoning.
    No training, no neural nets — pure strategy.
    """

    def act(self, obs: np.ndarray, valid_actions: list) -> int:
        if obs[OBS_PHASE] < 0.5:
            return self._bid(obs, valid_actions)
        return self._play(obs, valid_actions)

    # ──────────────────────────────────────────────────────────────
    # Bidding: expected-value estimation from compressed hand features
    # ──────────────────────────────────────────────────────────────

    def _bid(self, obs: np.ndarray, valid_actions: list) -> int:
        wizards   = obs[OBS_HAND_WIZARDS]   * 4.0   # actual wizard count in hand
        trump_cnt = obs[OBS_HAND_TRUMP]     * 10.0  # actual trump count
        high_cnt  = obs[OBS_HAND_HIGH]      * 10.0  # high non-trump count (ranks 10-13)
        max_trump = obs[OBS_HAND_MAX_TRUMP] * 13.0  # highest trump rank (0 if no trump)

        # Wizards are guaranteed tricks
        expected = wizards

        # Trump cards: strongest card wins near-certainly; others less so
        if trump_cnt > 0 and max_trump > 0:
            trump_quality = max_trump / 13.0          # 0.0 → 1.0
            best_trump_ev  = 0.40 + 0.55 * trump_quality   # ranges from 0.40 to 0.95
            extra_trump_ev = max(0.0, trump_cnt - 1.0) * 0.35
            expected += best_trump_ev + extra_trump_ev

        # High non-trump cards: conditional wins (~25 % of the time)
        expected += high_cnt * 0.25

        # Conservative bias: slight under-bid is marginally safer
        bid = int(round(expected - 0.15))

        # Clamp to valid range [0 … round_num]
        bid = max(valid_actions[0], min(valid_actions[-1], bid))
        return bid

    # ──────────────────────────────────────────────────────────────
    # Playing: context-aware priority rules
    # ──────────────────────────────────────────────────────────────

    def _play(self, obs: np.ndarray, valid_actions: list) -> int:
        tricks_needed    = obs[OBS_TRICKS_NEEDED] * 2.0    # un-normalise → integer
        position         = round(obs[OBS_POSITION] * 2.0)  # 0 = lead, 1 or 2 = follow

        trick_has_wizard = obs[OBS_TRICK_WIZARD] > 0.5
        trick_has_trump  = obs[OBS_TRICK_TRUMP]  > 0.5

        def avail(t: int) -> bool:
            return t in valid_actions

        def best(priority: list) -> int:
            for t in priority:
                if avail(t):
                    return t
            return valid_actions[0]   # safe fallback

        # ── Trick is already won by a Wizard — can't beat it ──────
        if trick_has_wizard:
            return best(DUMPING_ORDER)

        # ── Need to WIN more tricks ───────────────────────────────
        if tricks_needed > 0:
            if position == 0:
                # Leading: commit our strongest card to secure the trick
                return best(WINNING_ORDER)
            else:
                # Following: only invest if we can actually win
                if trick_has_trump:
                    # Trump is already on the table — only Wizard beats it
                    if avail(WIZARD):
                        return WIZARD
                    # Can't beat the trump; dump instead of wasting strong cards
                    return best(DUMPING_ORDER)
                else:
                    # No trump in trick yet — Wizard / Trump / High can win it
                    return best(WINNING_ORDER)

        # ── At or over bid — DUMP as cheaply as possible ─────────
        else:
            if position == 0:
                # Leading: play something that probably won't win
                return best(DUMPING_ORDER)
            else:
                # Following: avoid accidentally taking the trick
                # If trump is already in the trick, our non-trump low cards are safe
                # Either way, Jester > Low > High > Trump > Wizard minimises risk
                return best(DUMPING_ORDER)
