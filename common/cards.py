import random

SUITS = ['Red', 'Blue', 'Green', 'Yellow']


class Card:
    def __init__(self, suit, value):
        self.suit  = suit   # 'Red'/'Blue'/'Green'/'Yellow', or 'None' for specials
        self.value = value  # 1-13 normal, 0 = Jester, 14 = Wizard

    def __repr__(self):
        if self.value == 0:  return 'Jester'
        if self.value == 14: return 'Wizard'
        return f'{self.value} {self.suit}'


class Deck:
    def __init__(self):
        self.cards = []
        self._build()

    def _build(self):
        for suit in SUITS:
            for value in range(1, 14):
                self.cards.append(Card(suit, value))
        for _ in range(4):
            self.cards.append(Card('None', 0))   # Jester
            self.cards.append(Card('None', 14))  # Wizard

    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self):
        return self.cards.pop() if self.cards else None
