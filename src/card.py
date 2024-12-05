class Card:
    def __init__(self, value, suit, visible=False, bonus=False):
        """
        Initialize a card with given value, suit, visibility, and bonus status.

        Parameters:
        - value (int): The rank of the card (1 to 13, where 1 = Ace and 13 = King).
        - suit (int): The suit of the card (0 to 3).
        - visible (bool): Whether the card is face-up.
        - bonus (bool): Whether this card has been given a bonus for moving to the foundation.
        """
        self.value = value  # 1 to 13
        self.suit = suit  # 0 to 3
        # 0 - ♥ (hearts) (red)
        # 1 - ♦ (diamonds) (red)
        # 2 - ♣ (clubs) (black)
        # 3 - ♠ (spades) (black)
        # for observation:
        # 4 - unexisting suit for flatten()
        # 5 - invisible suit

        self.visible = visible  # Face-up or face-down
        self.bonus = bonus  # Bonus flag to prevent duplicate rewards

    def __repr__(self):
        visibility = "Visible" if self.visible else "Hidden"
        return f"Card(value={self.value}, suit={self.suit}, {visibility}, bonus={self.bonus})"

