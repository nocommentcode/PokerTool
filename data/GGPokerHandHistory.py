import re
from datetime import datetime
from enums.Card import Card
from enums.Value import Value
from enums.Suit import Suit


class GGPokerHandHistory:
    def __init__(self, text):
        self.num_players = self.extract_num_player(text)
        self.start_time = self.extract_start_time(text)
        self.dealer_pos = self.extract_dealer_pos(text)
        self.player_cards = list(reversed([self.extract_player_card(
            text, 0), self.extract_player_card(text, 1)]))
        self.table_cards = [self.extract_table_cards(text, 0),
                            self.extract_table_cards(text, 1),
                            self.extract_table_cards(text, 2),
                            self.extract_table_cards(text, 3),
                            self.extract_table_cards(text, 4)]

    def extract_num_player(self, text):
        seats = re.findall(r"Seat (\d):", text)
        return int(len(seats) / 2)

    def extract_start_time(self, text):
        timestamp = re.findall(r"\d{4}\/\d{2}\/\d{2} \d{2}:\d{2}:\d{2}", text)
        timestamp = timestamp[0]

        start_time = datetime.strptime(timestamp, '%Y/%m/%d %H:%M:%S')
        return start_time

    def extract_dealer_pos(self, text):
        all_seats = re.findall(r"Seat (?P<seat_no>\d):", text)

        my_seat = re.search(r"Seat (?P<seat_no>\d): Hero", text)
        my_seat = my_seat.group('seat_no')

        dealer_seat = re.search(r"Seat #(?P<seat_no>\d) is the button", text)
        dealer_seat = dealer_seat.group('seat_no')

        my_seat_idx = all_seats.index(my_seat)
        dealer_seat_idx = all_seats.index(dealer_seat)

        dealer_pos = dealer_seat_idx - my_seat_idx

        if dealer_pos < 0:
            # dealer is after me -> his seat - mine = dealer pos
            return self.num_players + dealer_pos

        # dealer is before me dealer pos = remaining seats + dealer pos
        return dealer_pos

    def extract_player_card(self, text, card_idx):
        player_cards = re.findall(
            r"Dealt to Hero \[(?P<card_1>..) (?P<card_2>..)\]", text)
        card_str = player_cards[0][card_idx]

        return self.string_to_card(card_str)

    def string_to_card(self, card_str):
        value = card_str[0]
        value = Value.from_string(value)

        suit = card_str[1]
        suit = Suit.from_string(suit.upper())

        return Card(suit, value)

    def extract_table_cards(self, text, card_idx):
        try:
            # flop
            if card_idx <= 2:
                cards = re.findall(
                    r"\*\*\* FLOP \*\*\* \[(?P<card_1>..) (?P<card_2>..) (?P<card_3>..)\]", text)
                card_str = cards[0][card_idx]

            # Turn
            if card_idx == 3:
                card = re.search(
                    r"\*\*\* TURN \*\*\* \[.. .. ..\] \[(?P<card>..)\]", text)
                card_str = card.group("card")

            # river
            if card_idx == 4:
                card = re.search(
                    r"\*\*\* RIVER \*\*\* \[.. .. .. ..\] \[(?P<card>..)\]", text)
                card_str = card.group("card")

            return self.string_to_card(card_str)
        except:
            return None


if __name__ == "__main__":
    with open('poker.txt') as f:
        text = f.read()
        hands = text.split('\n\n')
        for hand in hands:
            pokerHand = GGPokerHandHistory(hand)
