import random
import numpy as np
from common.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Baseline agent: always picks a random valid action."""

    def act(self, obs: np.ndarray, valid_actions: list) -> int:
        return random.choice(valid_actions)
