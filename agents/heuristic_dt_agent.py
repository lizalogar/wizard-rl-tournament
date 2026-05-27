import random
import numpy as np
from common.base_agent import (BaseAgent, OBS_PHASE, OBS_ROUND_NUM, 
                                OBS_HAND_WIZARDS, OBS_HAND_HIGH, OBS_HAND_TRUMP,
                                OBS_TRICKS_NEEDED, OBS_POSITION,
                                OBS_HAVE_WIZARD, OBS_HAVE_JESTER, OBS_HAVE_TRUMP)


class HeuristicDTAgent(BaseAgent):
    def __init__(self, name):
        super().__init__(name)

    def act(self, obs: np.ndarray, valid_actions: list) -> int:
        phase = obs[OBS_PHASE]

        if phase < 0.5:
            # ==========================================
            # BIDDING PHASE
            # ==========================================
            round_num   = round(obs[OBS_ROUND_NUM] * 10)
            wizards     = round(obs[OBS_HAND_WIZARDS] * 4)
            trump_cards = round(obs[OBS_HAND_TRUMP] * 10)
            
            # Translated from Bidding Heuristic Decision Tree
            if round_num <= 6:
                if trump_cards <= 2:
                    if wizards == 0:
                        action = 1
                    else:
                        action = 2
                else:
                    if trump_cards <= 3:
                        action = 3
                    else:
                        action = 0
            else:
                if trump_cards <= 2:
                    if wizards == 0:
                        action = 1
                    else:
                        action = 3
                else:
                    if trump_cards <= 4:
                        action = 4
                    else:
                        action = 0
                        
        else:
            # ==========================================
            # PLAYING PHASE
            # ==========================================
            # 0: wizard, 1: jester, 2: trump, 3: high, 4: low
            
            tricks_needed    = round(obs[OBS_TRICKS_NEEDED] * 2)
            position         = round(obs[OBS_POSITION] * 2)
            has_high_in_hand = bool(obs[OBS_HAND_HIGH] > 0.0)
            
            have_wizard_valid = bool(obs[OBS_HAVE_WIZARD] > 0.5)
            have_jester_valid = bool(obs[OBS_HAVE_JESTER] > 0.5)
            have_trump_valid  = bool(obs[OBS_HAVE_TRUMP] > 0.5)

            # Translated from Playing Heuristic Decision Tree
            # Note: Redundant sub-branches (where True/False yield the same class) have been simplified.
            if tricks_needed <= 0:
                # We don't want to win any more tricks!
                if not have_jester_valid:
                    action = 4  # Play low
                else:
                    if not has_high_in_hand:
                        if not have_trump_valid:
                            action = 1  # Play jester
                        else:
                            action = 4  # Play low
                    else:
                        if position == 0:
                            action = 1  # Play jester if we are leading the trick
                        else:
                            action = 4  # Play low
            else:
                # We NEED to win tricks!
                if not have_wizard_valid:
                    if not have_trump_valid:
                        if not has_high_in_hand:
                            action = 4  # Play low (nothing good to play)
                        else:
                            action = 3  # Play high
                    else:
                        action = 2  # Play trump
                else:
                    action = 0  # Play wizard!

        # Fallback: Decision trees might occasionally pick an invalid action 
        # because they are an approximation. Ensure we only play valid cards.
        if action not in valid_actions:
            action = random.choice(valid_actions)

        return action