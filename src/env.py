from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C, DQN
import gymnasium as gym
from gymnasium import spaces
import random
from rich.console import Console
from rich.text import Text
import numpy as np
import cv2
import os
from card import Card


# Initialize a Console object from the rich library for printing with styles
console = Console()

def card_to_one_hot(card):
    """
    Convert a card to a one-hot encoded array of length 54.
    """
    one_hot = [0] * 54
    if card.visible == True:  # Invisible card
        one_hot[53] = 1
    else:
        index = (card.value - 1) + (card.suit * 13)  # Calculate index for 52 cards
        one_hot[index] = 1
    return one_hot

class SolitaireEnv(gym.Env):
    def __init__(self):
        super(SolitaireEnv, self).__init__()
        self.tries = 400
        # The action space now includes three parts: action type, source column, destination column
        # Action type: 0 (Move within tableau), 1 (Draw card), 2 (Move to foundation); Source column (0-11) and card index (0-18); Destination column (0-6)
        self.action_space = spaces.MultiDiscrete([3, 12, 18, 7])

        # Define observation space with structured tableau, foundation, and draw pile
        self.observation_space = spaces.Dict({
            'tableau': spaces.MultiDiscrete([2] * 54 * 7 * 18),  # Each card is one-hot (54), 7 columns, 18 cards max
            'foundation': spaces.MultiDiscrete([14, 14, 14, 14]),  # Foundation unchanged
            'top_card': spaces.MultiDiscrete([2] * 54),  # One-hot encoded top card
        })

        # Track revealed cards from the draw pile
        self.revealed_cards = []
        self.draw_index = 0  # Tracks current position in draw pile

        self.tableau = None
        self.foundation = None
        self.draw_pile = None
        self.draw_pile_cycles = 3
        self.done = False
        self.reward = 0
        self.colors = {
            0: "red",
            1: "red",
            2: "black",
            3: "black"
        }
        self.king_value = 13
        self.ace_value = 1
        self.invisible_value = 14
        self.unexisting_value = 0
        self.invisible_suit = 5
        self.unexisting_suit = 4
        self._reset_game_state()

    def _reset_game_state(self):
        # Initialize the deck as a list of Card objects
        deck = [Card(value, suit) for suit in range(4) for value in range(self.ace_value, self.king_value+1)]
        random.shuffle(deck)

        # Initialize tableau with some cards face-down
        self.tableau = [[deck.pop() for _ in range(i + 1)] for i in range(7)]
        for col in self.tableau:
            for card in col[:-1]:
                card.visible = False  # Face-down
            col[-1].visible = True  # Top card face-up

        # Foundation starts empty
        self.foundation = [[] for _ in range(4)]
        self.draw_pile = deck  # Remaining cards in the draw pile
        self.done = False
        self.reward = 0
        self.revealed_cards = []
        self.draw_index = 0

    def reset(self, seed=None):
        """Resets the environment to the initial state."""
        self._reset_game_state()
        return self._get_observation(), {}

    def is_valid_tableau_move(self, source_col, source_idx):
        """Custom logic to validate moves within tableau based on game rules."""
        # Example logic: check if there's a card at the specified column and index
        if source_col >= len(self.tableau):
            return False
        if source_idx >= len(self.tableau[source_col]):
            return False
        if not self.tableau[source_col][source_idx].visible:
            return False
        return True

    def is_valid_move_to_foundation(self, source_col):
        """Custom logic to validate moves to the foundation."""
        # Example logic: check if a card from the source column can move to the foundation
        if source_col >= len(self.tableau):
            return False
        if len(self.tableau[source_col]) == 0:
            return False
        return True

    def compute_action_mask(self):
        # Initialize mask with zeros (invalid actions by default)
        mask = np.zeros((3, 12, 18, 7), dtype=int)

        # Iterate over all possible actions
        for action_type in range(3):  # Action type: 0, 1, 2
            for source_col in range(12):  # Source column: 0-11
                for source_idx in range(18):  # Source index: 0-17
                    for dest_col in range(7):  # Destination column: 0-6
                        if action_type == 0:  # Move within tableau
                            if 0 <= source_col <= 6:
                                # Check if this source_col and source_idx are valid
                                if self.is_valid_tableau_move(source_col, source_idx):
                                    mask[0, source_col, source_idx, dest_col] = 1
                            elif 7 <= source_col <= 11:
                                # Ignore source_idx for these columns, but only allow source_idx = 0
                                if self.is_valid_tableau_move(source_col, 0):
                                    mask[0, source_col, 0, dest_col] = 1

                        elif action_type == 2:  # Move to foundation
                            # Only source_col is relevant; ignore source_idx and dest_col
                            if self.is_valid_move_to_foundation(source_col):
                                mask[2, source_col, 0, 0] = 1

        mask[1, 0, 0, 0] = 1   # only valid move for drawing card

        return mask.flatten()


    def _get_observation(self):
    # Convert tableau to one-hot encoding
        tableau_observation = [
            [card_to_one_hot(card) for card in column]
            for column in self.tableau
        ]

        # Pad tableau columns to a fixed length (18 cards)
        max_length = 18
        tableau_observation_padded = [
            column + [[0] * 54] * (max_length - len(column))
            for column in tableau_observation
        ]

        # Flatten the tableau for observation
        tableau_array = np.array(tableau_observation_padded).reshape(-1)

        # Foundation observation (unchanged)
        foundation_observation = [len(pile) for pile in self.foundation]

        # Top card observation
        if self.revealed_cards:
            top_card = self.revealed_cards[-1]
            top_card_observation = card_to_one_hot(top_card)
        else:
            top_card_observation = [0] * 54
            top_card_observation[52] = 1  # Non-existent card

        # Combine all parts of the observation
        result = {
            "tableau": tableau_array,
            "foundation": np.array(foundation_observation),
            "top_card": np.array(top_card_observation),
        }
        return result


    def step(self, action: list):
        # action: [int, list[int,int], int]
        # action_type: int
        # source: list[int,int] - [column, card in column]. Column numbers:
        # 0-6: tableau
        # 7-10: foundation
        # 11: draw pile
        # destination: int - column, no need  since all the cards will be moved on top of it

        action_type, source1, source2, destination = action
        current_reward = -1   # Base penalty for each action

        if action_type == 0:  # Move Card within Tableau
            valid_move_made, reward = self._move_within_tableau([source1, source2], destination)   # here reward should be negative
            current_reward += reward
            if not valid_move_made:
                #print("The move isn't valid")
                current_reward -= 10  # Extra penalty for invalid move

        elif action_type == 1:  # Draw Card from Draw Pile
            empty = self._draw_card()
            current_reward -= 70
            if empty:
                current_reward -= 100

        elif action_type == 2:  # Move Card to Foundation
            valid_move_made, reward = self._move_to_foundation(source1)
            current_reward += reward

        flipped_count = self._flip_visible_cards()
        current_reward += flipped_count * 1600

        # Check if game is won (all foundations complete)
        if all(len(foundation) == 13 for foundation in self.foundation):
            self.done = True

        self.reward += current_reward
        self.tries -= 1

        return self._get_observation(), current_reward, self.done, self.tries <= 0, {}  # False - truncated field (hz zachem), {} - info field (tozhe hz zachem)

    def _draw_card(self):
        # Reveal 1 card at a time from the draw pile
        if self.draw_pile:
            card = self.draw_pile.pop()
            card.visible = True
            self.revealed_cards.append(card)
            return False
        else:
            self.draw_pile = self.revealed_cards[::-1]  # Restart the draw pile if we reach the end
            self.revealed_cards = []
            self.draw_pile_cycles-=1
            if self.draw_pile_cycles < 0:
                return True


    def _is_alternating_color(self, object1, object2):
        # Check if the objects have alternating colors
        return self.colors[object1] != self.colors[object2]


    # all return numbers after false are negative, and positive after true
    def _move_within_tableau(self, source: list[int], destination: int):
        if destination > 6 or destination < 0:
            #print("Wrong destination column")
            return False, -10
        # If the source is from the draw pile
        if source[0] == 11:
            if not self.revealed_cards:
                #print("Invalid move: No cards revealed in the draw pile")
                return False, -50 # No cards revealed in the draw pile
            # Use the last revealed card from the draw pile
            card_to_move = self.revealed_cards[-1]

            # Check if destination column is empty (only Kings can move to empty columns)
            if not self.tableau[destination]:
                if card_to_move.value == self.king_value: # King card value
                    self.tableau[destination].append(card_to_move)
                    self.revealed_cards.pop()  # Remove from revealed list
                    return True, 500
                else:
                    #print("Invalid move: Only Kings can move to an empty column")
                    return False, -60  # Only Kings can move to an empty column

            # Check if the move is valid based on the destination column's top card
            dest_card = self.tableau[destination][-1]
            if (card_to_move.value == dest_card.value - 1 and
                self._is_alternating_color(card_to_move.suit, dest_card.suit)):
                self.tableau[destination].append(card_to_move)
                self.revealed_cards.pop()  # Remove from revealed list
                return True, 500

            #print("Invalid move: Invalid move for draw pile card")
            return False, -50  # Invalid move for draw pile card

        if 7 <= source[0] <= 10:
            suit = source[0] - 7
        if self.foundation[suit] and self.tableau[destination]:
            card_to_move = self.foundation[suit][-1]
            dest_card = self.tableau[destination][-1]
            if (card_to_move.value == dest_card.value - 1 and
                self._is_alternating_color(card_to_move.suit, dest_card.suit)):
                self.tableau[destination].append(card_to_move)
                self.foundation[suit].pop()  # Remove from revealed list
                return True, 500
            return False, -50
        else:
            return False, -50

        if source[0] < 0:
            #print("Invalid move: Wrong column number")
            return False, -100
        if source[1] >= len(self.tableau[source[0]]):
            #print("Invalid move: Wrong card index number")
            return False, -100

        card_column = source[0]
        card_index = source[1]
        cards_to_move = self.tableau[card_column][card_index:]

        # Check if destination column is empty (only Kings can be moved to an empty column)
        if not self.tableau[destination]:
            if cards_to_move[0].value == self.king_value: # King card value
                # Move the sequence
                self.tableau[destination].extend(cards_to_move)
                del self.tableau[card_column][card_index:]
                return True, 500
            else:
                #print("Invalid move: Only Kings can be moved to an empty column")
                return False, -60  # Only Kings can be moved to an empty column

        # Check if the move is valid based on the destination column’s top card
        dest_card = self.tableau[destination][-1]
        if (cards_to_move[0].value == dest_card.value - 1 and
                self._is_alternating_color(cards_to_move[0].suit, dest_card.suit)):
            # Move the sequence
            self.tableau[destination].extend(cards_to_move)
            del self.tableau[card_column][card_index:]
            return True, 500

        #print("Invalid move: Invalid move within tableau")
        return False, -40  # Move was invalid


    def _move_to_foundation(self, source): # source is int, since we move the top card of source to top of foundation
        valid_reward = 130
        # If source is the draw pile (denoted by 11), take the last revealed card
        if source == 11:
            if not self.revealed_cards:
                #print("Invalid move: No revealed cards in draw pile")
                return False, -50  # No revealed cards in draw pile
            card = self.revealed_cards[-1]
            foundation_index = card.suit # Determine foundation based on suit

            if len(self.foundation[foundation_index]) == card.value - self.ace_value:  # -ace_value because card values are from 2 to 14
                # Move the card to the foundation and remove from revealed list
                self.foundation[foundation_index].append(card)
                self.revealed_cards.pop()
                if not card.bonus:
                    card.bonus = True
                return True, valid_reward

            #print("Invalid move: Invalid move to foundation from draw pile")
            return False, -40  # Invalid move

        # Validate source column
        if source < 0 or source > 6 or not self.tableau[source]:
            #print("Invalid move: No card to move")
            return False, -60  # Invalid move, no card to move

        # Get the top card from the source column
        card = self.tableau[source][-1]
        foundation_index = card.suit  # Determine foundation pile based on suit

        # Check if the card can move to the foundation (must be in ascending order)
        if len(self.foundation[foundation_index]) == card.value - self.ace_value:     # -ace_value because card values are from 2 to 14
            # Move the card to the foundation and remove from tableau
            self.foundation[foundation_index].append(self.tableau[source].pop())
            if not card.bonus:
                card.bonus = True
            return True, valid_reward

        #print("Invalid move: Invalid move to foundation from tableau")
        return False, -40  # Move was invalid


    def _flip_visible_cards(self):
        flipped_count = 0
        for column in self.tableau:
            if column and not column[-1].visible:  # If the top card is face-down
                column[-1].visible = True  # Flip it face-up
                flipped_count += 1
        return flipped_count


    def render(self, mode='human'):
        # Foundations
        foundation_str = []
        for pile in self.foundation:
            if pile:
                card = pile[-1]
                suit = ['[bold red] ♥[/bold red]', '[bold red] ♦[/bold red]', ' ♣', ' ♠'][card.suit]
                # Apply color red for red suits (Diamonds and Hearts)
                foundation_str.append(f"| {card.value} {suit} |" if card.value != self.invisible_value else f"| A {suit} |")
            else:
                foundation_str.append("|     |")  # Empty foundation pile

        # Print foundation row
        console.print("Foundations:", "  ".join(foundation_str))

        # Tableau - display all 7 columns in a single row
        tableau_str = []
        for col in self.tableau:
            tableau_str.append(" ".join([f"┌─────┐" if card.visible else "┌─────┐" for card in col]))  # Card tops
            tableau_str.append(" ".join([f"| {card.value if card.visible else ' ?'}{' ' if card.visible and len(str(card.value)) != 2 else ''}{['[bold red] ♥[/bold red]', '[bold red] ♦[/bold red]', ' ♣', ' ♠'][card.suit] if card.visible else '  '}|" for card in col]).replace(" 0 ", " A "))  # Card values
            tableau_str.append(" ".join([f"|     |" for _ in col]))  # Empty space for spacing between cards
            tableau_str.append(" ".join([f"└─────┘" for _ in col]))  # Card bottoms

        # Print tableau columns in one row
        tableau_str = '\n'.join(tableau_str)
        console.print("Tableau:\n" + tableau_str)

        # Draw pile (remaining count, last 3 revealed, discarded count)
        draw_pile_display = f"Draw Pile: {len(self.draw_pile)} cards remaining"

        # Last 3 revealed cards (if any)
        last_three = [f"|{card.value if card.visible else ' ?'}{' ' if card.visible and len(str(card.value)) != 2 else ''}{['[bold red] ♥[/bold red]', '[bold red] ♦[/bold red]', ' ♣', ' ♠'][card.suit] if card.visible else '  '}|" for card in self.revealed_cards[-3:]]
        last_three_display = f"\nLast 3 Drawn: {' '.join(last_three)}"

        # Discarded cards (number of discarded cards)

        # Print the full draw pile row
        console.print(draw_pile_display, last_three_display)
