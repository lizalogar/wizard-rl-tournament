"""
WizardGame engine.

DO NOT MODIFY THIS FILE.  Shared by all agents.

Usage:
    game = WizardGame([agent_a, agent_b, agent_c])
    scores = game.play_episode()   # plays 10 rounds, returns [score_a, score_b, score_c]

The engine owns all game state (hands, scores, bids, trump card).
Agents only receive an observation vector and return an action index.
"""

import inspect
import random
import numpy as np
from .cards import Deck
from .base_agent import (
    BaseAgent, OBS_SIZE,
    OBS_ROUND_NUM, OBS_PHASE, OBS_MY_BID,
    OBS_TRICKS_WON, OBS_TRICKS_NEEDED, OBS_POSITION,
    OBS_HAND_WIZARDS, OBS_HAND_HIGH, OBS_HAND_TRUMP,
    OBS_HAVE_WIZARD, OBS_HAVE_JESTER, OBS_HAVE_TRUMP,
    OBS_TRICK_WIZARD, OBS_TRICK_TRUMP,
    OBS_HAND_MAX_TRUMP, OBS_OPP_MAX_NEED, OBS_OPP_MIN_NEED,
    CARD_TYPES,
)


class PlayerState:
    """Mutable per-player state owned by the engine."""
    def __init__(self, name: str):
        self.name        = name
        self.hand        = []
        self.tricks_won  = 0
        self.current_bid = 0
        self.total_score = 0
        self._prev_score = 0


class WizardGame:
    def __init__(self, agents: list, verbose: bool = False):
        self.agents     = agents
        self.states     = [PlayerState(a.name) for a in agents]
        self.round_num  = 1
        self.num_rounds = 60 // len(agents)   # 3p→20, 4p→15, 5p→12, 6p→10
        self.verbose    = verbose
        self.trump_card          = None
        self.current_trick_cards = []   # cards played so far this trick

        # Check once which agents accept the trick_card_types argument
        self._supports_trick_cards = [
            len(inspect.signature(a.on_trick_result).parameters) >= 3
            for a in agents
        ]

        # Check once which agents want the card-groups dict passed to act()
        self._supports_groups = [
            len(inspect.signature(a.act).parameters) >= 3
            for a in agents
        ]

        # How many obs features each agent expects (defaults to OBS_SIZE for agents that don't declare obs_size)
        self._agent_obs_size = [getattr(a, 'obs_size', OBS_SIZE) for a in agents]

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def play_episode(self) -> list:
        """
        Play a full game (num_rounds rounds).
        """
        for ps in self.states:
            ps.total_score = 0

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  WIZARD  |  {len(self.agents)} players  |  {self.num_rounds} rounds")
            print(f"{'='*60}")

        for round_num in range(1, self.num_rounds + 1):
            self.round_num = round_num
            # Initialize exact card memory perfectly before each round
            self._burned_cards = np.zeros(60, dtype=np.float32)
            self.play_round()

        if self.verbose:
            print(f"\n{'='*60}  FINAL SCORES")
            for ps in self.states:
                print(f"  {ps.name:<12}  {ps.total_score:>6}")
            winner = max(self.states, key=lambda s: s.total_score)
            print(f"  Winner: {winner.name}")
            print(f"{'='*60}\n")

        for agent in self.agents:
            agent.on_episode_end()

        return [ps.total_score for ps in self.states]

    # ------------------------------------------------------------------
    # Round
    # ------------------------------------------------------------------

    def play_round(self):
        self._setup_round()

        n          = len(self.agents)
        trump_suit = self._trump_suit()
        start_idx  = (self.round_num - 1) % n

        if self.verbose:
            trump_str = repr(self.trump_card) if self.trump_card else 'None (no trump)'
            print(f"\n{'─'*64}")
            print(f"  Round {self.round_num:>2}/{self.num_rounds}   Trump: {trump_str}")
            print(f"{'─'*64}")
            for ps in self.states:
                groups = self._group_by_type(ps.hand, trump_suit)
                parts  = []
                for t in CARD_TYPES:
                    if t in groups:
                        cards_str = ' '.join(
                            repr(c) for c in sorted(groups[t], key=lambda c: c.value)
                        )
                        parts.append(f'{t}[{cards_str}]')
                print(f"  {ps.name:<12} {',  '.join(parts)}")

        # --- Bidding ---
        for i in range(n):
            idx   = (start_idx + i) % n
            agent = self.agents[idx]
            is_advanced = getattr(agent, 'uses_exact_cards', False)
            
            valid = list(range(self.round_num + 1))   # bids 0..round_num
            
            if is_advanced:
                obs = self._build_exact_obs(self.states[idx], [], self.round_num, phase=0.0)
                bid = int(agent.act(obs, valid))
            else:
                obs = self._build_obs(idx, phase=0)[:self._agent_obs_size[idx]]
                bid = int(agent.act(obs, valid))
                
            self.states[idx].current_bid = bid

        if self.verbose:
            print()
            for i in range(n):
                ps = self.states[i]
                print(f"  Bid  {ps.name:<12} → {ps.current_bid}")

        # --- Tricks ---
        lead_idx = start_idx
        for trick_num in range(self.round_num):
            trick_cards   = []
            trick_indices = []
            lead_suit     = None

            if self.verbose:
                print(f"\n  Trick {trick_num + 1}/{self.round_num}")

            for i in range(n):
                idx   = (lead_idx + i) % n
                ps    = self.states[idx]
                agent = self.agents[idx]
                is_advanced = getattr(agent, 'uses_exact_cards', False)

                self.current_trick_cards = trick_cards

                if is_advanced:
                    # ====== SMART AGENT LOGIC ======
                    obs = self._build_exact_obs(ps, trick_cards, self.round_num, phase=1.0)
                    legal_mask = self._get_exact_mask(ps.hand, lead_suit)
                    
                    chosen_card_id = agent.act(obs, legal_mask)
                    
                    # Find and pop that exact card from the hand
                    card = next(c for c in ps.hand if getattr(c, 'id', -1) == chosen_card_id)
                    ps.hand.remove(card)
                else:
                    # ====== LEGACY AGENT LOGIC ======
                    valid = self._get_valid_cards(ps.hand, lead_suit)
                    groups     = self._group_by_type(valid, trump_suit)
                    type_valid = [CARD_TYPES.index(t) for t in groups]

                    obs    = self._build_obs(idx, phase=1, valid_cards=valid)[:self._agent_obs_size[idx]]
                    if self._supports_groups[idx]:
                        action = int(agent.act(obs, type_valid, groups))
                    else:
                        action = int(agent.act(obs, type_valid))

                    chosen_type = CARD_TYPES[action]
                    pick = getattr(agent, 'pick_card', None)
                    card = pick(groups[chosen_type]) if pick else random.choice(groups[chosen_type])
                    ps.hand.remove(card)

                if self.verbose:
                    print(f"    {ps.name:<12} plays {repr(card)}")

                trick_cards.append(card)
                trick_indices.append(idx)
                
                # Remember exactly which card was burned
                self._burned_cards[getattr(card, 'id', 0)] = 1.0

                if lead_suit is None and card.value != 0:
                    lead_suit = card.suit

            win_pos = self._evaluate_trick(trick_cards)
            win_idx = trick_indices[win_pos]
            self.states[win_idx].tricks_won += 1

            if self.verbose:
                print(f"    => {self.states[win_idx].name} wins")

            # Deliver per-trick feedback to every agent in the trick
            trick_card_types = [
                CARD_TYPES.index(self._card_type(c, trump_suit))
                for c in trick_cards
            ]
            for idx in trick_indices:
                ps           = self.states[idx]
                won          = (idx == win_idx)
                still_needed = ps.current_bid - ps.tricks_won
                if won:
                    trick_reward = 15 if still_needed >= 0 else -15
                else:
                    trick_reward = 10 if still_needed <= 0 else -5
                if self._supports_trick_cards[idx]:
                    self.agents[idx].on_trick_result(won, trick_reward, trick_card_types)
                else:
                    self.agents[idx].on_trick_result(won, trick_reward)

            lead_idx = win_idx

        self._calculate_scores()

        if self.verbose:
            print()
            for ps in self.states:
                delta    = ps.total_score - ps._prev_score
                outcome  = 'exact' if ps.tricks_won == ps.current_bid else 'MISS'
                sign     = '+' if delta >= 0 else ''
                print(f"  {ps.name:<12} {ps.tricks_won}/{ps.current_bid} [{outcome}]"
                      f"  {sign}{delta:>4}  total: {ps.total_score}")

        for idx, agent in enumerate(self.agents):
            reward = self.states[idx].total_score - self.states[idx]._prev_score
            agent.on_round_end(reward)


    # ------------------------------------------------------------------
    # Advanced Observation Helpers
    # ------------------------------------------------------------------

    def _build_exact_obs(self, ps, trick_cards, round_number, phase) -> np.ndarray:
        """Builds a 186-element array for advanced PPO agents."""
        # 1. Game State (6 floats)
        state = np.array([
            round_number       / self.num_rounds,
            phase,
            ps.current_bid     / self.num_rounds,
            ps.tricks_won      / self.num_rounds,
            max(-2.0, min(2.0, ps.current_bid - ps.tricks_won)) / 2.0,
            len(trick_cards)   / 2.0,
        ], dtype=np.float32)
        
        # 2. Exact Hand (60 floats)
        hand_arr = np.zeros(60, dtype=np.float32)
        for c in ps.hand:
            hand_arr[getattr(c, 'id', 0)] = 1.0
            
        # 3. Current Trick Context (60 floats)
        trick_arr = np.zeros(60, dtype=np.float32)
        for c in trick_cards:
            trick_arr[getattr(c, 'id', 0)] = 1.0
            
        # 4. Perfect Memory - cards burned this round (60 floats)
        return np.concatenate([state, hand_arr, trick_arr, self._burned_cards])

    def _get_exact_mask(self, hand, lead_suit) -> np.ndarray:
        """Returns a 60-element array where 1.0 means the exact card is legal to play."""
        mask = np.zeros(60, dtype=np.float32)
        has_lead_suit = any(c.suit == lead_suit for c in hand) if lead_suit else False
        
        for c in hand:
            c_id = getattr(c, 'id', 0)
            if c.value == 0 or c.value == 14:  # Jesters and Wizards always legal
                mask[c_id] = 1.0
            elif lead_suit and has_lead_suit:
                if c.suit == lead_suit:
                    mask[c_id] = 1.0
            else:
                mask[c_id] = 1.0
                
        return mask

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_round(self):
        deck = Deck()
        deck.shuffle()

        for ps in self.states:
            ps.hand        = []
            ps.tricks_won  = 0
            ps.current_bid = 0
            ps._prev_score = ps.total_score

        for _ in range(self.round_num):
            for ps in self.states:
                ps.hand.append(deck.draw())

        self.trump_card = deck.draw() if deck.cards else None

    def _trump_suit(self):
        if self.trump_card and self.trump_card.suit != 'None':
            return self.trump_card.suit
        return None

    # ------------------------------------------------------------------
    # Card helpers
    # ------------------------------------------------------------------

    def _card_type(self, card, trump_suit):
        if card.value == 14:                            return 'wizard'
        if card.value == 0:                             return 'jester'
        if trump_suit and card.suit == trump_suit:      return 'trump'
        if card.value >= 10:                            return 'high'
        return 'low'

    def _get_valid_cards(self, hand, lead_suit):
        if lead_suit is None:
            return list(hand)
        matching = [c for c in hand
                    if c.suit == lead_suit and c.value not in (0, 14)]
        if matching:
            return [c for c in hand if c.suit == lead_suit or c.value in (0, 14)]
        return list(hand)

    def _group_by_type(self, cards, trump_suit):
        groups = {}
        for c in cards:
            t = self._card_type(c, trump_suit)
            groups.setdefault(t, []).append(c)
        return groups

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_obs(self, agent_idx: int, phase: int, valid_cards=None) -> np.ndarray:
        ps         = self.states[agent_idx]
        trump_suit = self._trump_suit()

        num_wizards = sum(1 for c in ps.hand if c.value == 14)
        num_high    = sum(1 for c in ps.hand if 10 <= c.value <= 13)
        num_trump   = sum(1 for c in ps.hand
                         if trump_suit and c.suit == trump_suit)

        tricks_needed = max(-2.0, min(2.0, ps.current_bid - ps.tricks_won))
        position      = min(len(self.current_trick_cards), 2)

        vc          = valid_cards or []
        have_wizard = float(any(c.value == 14 for c in vc))
        have_jester = float(any(c.value == 0  for c in vc))
        have_trump  = float(trump_suit is not None
                            and any(c.suit == trump_suit for c in vc))

        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        obs[OBS_ROUND_NUM]     = self.round_num    / self.num_rounds
        obs[OBS_PHASE]         = float(phase)          # 0 = bid, 1 = play
        obs[OBS_MY_BID]        = ps.current_bid   / self.num_rounds
        obs[OBS_TRICKS_WON]    = ps.tricks_won    / self.num_rounds
        obs[OBS_TRICKS_NEEDED] = tricks_needed  / 2
        obs[OBS_POSITION]      = position       / 2
        obs[OBS_HAND_WIZARDS]  = min(num_wizards, 4)  / 4
        obs[OBS_HAND_HIGH]     = min(num_high,   10)  / 10
        obs[OBS_HAND_TRUMP]    = min(num_trump,  10)  / 10
        obs[OBS_HAVE_WIZARD]   = have_wizard
        obs[OBS_HAVE_JESTER]   = have_jester
        obs[OBS_HAVE_TRUMP]    = have_trump
        # Cards already played in the current trick (before this agent acts)
        obs[OBS_TRICK_WIZARD]  = float(any(c.value == 14 for c in self.current_trick_cards))
        obs[OBS_TRICK_TRUMP]   = float(
            trump_suit is not None and
            any(c.suit == trump_suit and c.value not in (0, 14)
                for c in self.current_trick_cards)
        )

        # Highest trump rank in hand — tells agent how strong its best trump is
        if trump_suit:
            trump_ranks = [c.value for c in ps.hand
                           if c.suit == trump_suit and c.value not in (0, 14)]
            obs[OBS_HAND_MAX_TRUMP] = (max(trump_ranks) / 13.0) if trump_ranks else 0.0
        else:
            obs[OBS_HAND_MAX_TRUMP] = 0.0

        # Opponent bid pressure: max and min of (bid - tricks_won) across all opponents
        opp_needs = [
            max(-2.0, min(2.0, float(s.current_bid - s.tricks_won)))
            for i, s in enumerate(self.states) if i != agent_idx
        ]
        obs[OBS_OPP_MAX_NEED] = (max(opp_needs) / 2.0) if opp_needs else 0.0
        obs[OBS_OPP_MIN_NEED] = (min(opp_needs) / 2.0) if opp_needs else 0.0

        return obs

    # ------------------------------------------------------------------
    # Trick evaluation
    # ------------------------------------------------------------------

    def _evaluate_trick(self, trick_cards: list) -> int:
        """Return the index (0-based) of the winning card in trick_cards."""
        # First Wizard played wins
        for i, card in enumerate(trick_cards):
            if card.value == 14:
                return i

        # Find lead suit (first non-Jester)
        lead_suit = None
        for card in trick_cards:
            if card.value != 0:
                lead_suit = card.suit
                break

        if lead_suit is None:
            return 0  # all Jesters — first player wins

        trump_suit = self._trump_suit()

        # Highest trump wins if any trump was played
        if trump_suit:
            best_val, best_idx = -1, -1
            for i, card in enumerate(trick_cards):
                if card.suit == trump_suit and card.value not in (0, 14):
                    if card.value > best_val:
                        best_val, best_idx = card.value, i
            if best_idx != -1:
                return best_idx

        # Highest card of lead suit wins
        best_val, best_idx = -1, -1
        for i, card in enumerate(trick_cards):
            if card.suit == lead_suit and card.value not in (0, 14):
                if card.value > best_val:
                    best_val, best_idx = card.value, i
        return best_idx

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _calculate_scores(self):
        for ps in self.states:
            if ps.tricks_won == ps.current_bid:
                ps.total_score += 20 + 10 * ps.tricks_won
            else:
                ps.total_score -= abs(ps.tricks_won - ps.current_bid) * 10