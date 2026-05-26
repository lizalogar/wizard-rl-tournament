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
        card_id = 0
        
        # 0 to 51: Standard colored cards
        for suit in SUITS:
            for value in range(1, 14):
                c = Card(suit, value)
                c.id = card_id
                self.cards.append(c)
                card_id += 1
                
        # 52 to 55: The 4 Jesters
        for _ in range(4):
            c = Card('None', 0)
            c.id = card_id
            self.cards.append(c)
            card_id += 1
            
        # 56 to 59: The 4 Wizards
        for _ in range(4):
            c = Card('None', 14)
            c.id = card_id
            self.cards.append(c)
            card_id += 1

    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self):
        return self.cards.pop() if self.cards else None
