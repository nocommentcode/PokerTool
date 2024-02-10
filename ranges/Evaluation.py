import math
from enums.Card import Card
from enums.Value import Value
from enums.Hand import Hand
from enums.Suit import Suit
from ranges.Deck import Deck
from ranges.EvalutionResult import EvalutionResult
from ranges.Flush import Flush
from ranges.FourOfAKind import FourOfAKind
from ranges.FullHouse import FullHouse
from ranges.HighCard import HighCard
from ranges.Pair import Pair
from ranges.Straight import Straight
from ranges.StraightFlush import StraightFlush
from ranges.ThreeOfAKind import ThreeOfAKind
from ranges.TwoPair import TwoPair


import numpy as np


from typing import List


class Evaluation:
    def __init__(self, hand: Hand, table_cards: List[Card], player_count: int):
        self.player_cards = np.array([
            self.card_to_num(card) for card in hand.cards()])
        self.table_cards = np.array(
            [self.card_to_num(card) for card in table_cards])

        self.player_count = player_count

    @staticmethod
    def card_to_num(card: Card):
        def handle_card_value(value):
            # shift 1 down (2 = 1 in Card, should be 1 for this)
            int_value = value.value - 1
            # handle ace
            if int_value == 0:
                int_value = 13

            return int_value

        card_num, card_suit = handle_card_value(
            card.value), card.suit.value - 1

        if card_suit == 0:
            return card_num * 4

        else:
            return (card_num - 1) * 4 + card_suit

    @staticmethod
    def num_to_card(num: int):
        value, suit = math.ceil(num / 4), num % 4
        if value == 13:
            value = 1
        else:
            value += 1
        return Card(Suit.from_index(int(suit + 1)),
                    Value.from_index(int(value)))

    def print_debug(self, player_cards, num_iters, winners, num_players=None):
        player = player_cards.shape[2] if num_players is None else num_players
        for iter in range(num_iters):
            winner = np.where(winners[iter])[0]
            print(
                f"Iteration {iter+1}: {' '.join([('me' if w == 0 else str(w)) for w in winner])}")

            for i in range(player):
                label = "me: " if i == 0 else f"{i}: "
                cards = [str(self.num_to_card(card))
                         for card in player_cards[iter, :, i]]
                print(
                    f"{label}{' '.join(cards)}")
            print("\n")

    def deal_cards(self, iterations, deck: Deck):
        player_hands = np.zeros((iterations, 7, self.player_count))
        for i in range(self.player_count):
            # player hole cards
            if i == 0:
                player_hands[:, :2, i] = self.player_cards

            # opponent hole cards
            else:
                player_hands[:, :2, i] = deck.deal_player_cards(i-1)

        # table cards already delt
        player_hands[:, 2:2+len(self.table_cards),
                     :] = self.table_cards[None, :, None]

        # random remaining table cards
        cards_to_deal = 5 - len(self.table_cards)
        player_hands[:, 2+len(self.table_cards):,
                     :] = deck.deal_table_cards(cards_to_deal)[:, :, None]

        cards = np.ceil(player_hands / 4)
        suits = player_hands % 4 * 1

        # shape = (iterations, card, player)
        return cards, suits, player_hands

    @staticmethod
    def get_card_counts(player_cards):
        # shape = (iters x players x card)
        return (np.arange(13, 0, -1) == player_cards[:, :, :, None]).sum(1)

    @staticmethod
    def get_of_a_kinds(card_counts):
        cards_13_to_1 = np.arange(13, 0, -1)

        # shape = (iters x player x pair_idx)
        all_pairs = np.sort((card_counts == 2) * cards_13_to_1, axis=2)
        pairs = all_pairs[:, :, ::-1][:, :, :3]

        # shape = (iters x player x three_idx)
        all_threes = np.sort((card_counts == 3) * cards_13_to_1, axis=2)
        threes = all_threes[:, :, ::-1][:, :, :2]

        # shape = (iters x players x 1)
        all_fours = np.sort((card_counts == 4) * cards_13_to_1, axis=2)
        fours = all_fours[:, :, ::-1][:, :, 0]

        # shape = (iters x players x 5)
        all_kickers = np.sort((card_counts == 1) * cards_13_to_1, axis=2)
        kickers = all_kickers[:, :, ::-1][:, :, :5]

        return pairs, threes, fours, kickers

    def run(self, deck, iterations=10000, debug=False):
        cards, suits, player_hands = self.deal_cards(iterations, deck)

        card_counts = self.get_card_counts(cards)
        pairs, threes, fours, kickers = self.get_of_a_kinds(card_counts)

        evaluations = [StraightFlush(cards, suits),
                       FourOfAKind(fours, kickers),
                       FullHouse(pairs, threes),
                       Flush(cards, suits),
                       Straight(card_counts),
                       ThreeOfAKind(threes, pairs, kickers),
                       TwoPair(pairs, threes, kickers),
                       Pair(pairs, threes, kickers)]
        evaluations.append(HighCard(evaluations, kickers))

        winners = self.get_results(evaluations)
        if debug:
            self.print_debug(player_hands, 30, winners)

        wins, draws = self.calc_wins_draws(winners)
        equity = wins / iterations
        win_percent = (wins + draws) / iterations

        return EvalutionResult(evaluations, equity, win_percent)

    def get_results(self, evalutions):
        multipliers = np.array([eval.multiplier for eval in evalutions])
        hits = np.stack([eval.hit for eval in evalutions], axis=0)
        scores = np.stack([eval.score for eval in evalutions], axis=0)

        final_scores = multipliers[:, None, None] * hits * scores
        final_scores = np.sort(final_scores, axis=0)[::-1, :, :]

        winners = (final_scores[0, ::] == np.amax(
            final_scores[0, :, :], axis=1)[:, None])
        return winners

    def calc_wins_draws(self, winners):
        num_player_non_loss = winners[:, 0].sum()

        player_winner_mask = np.zeros(self.player_count, dtype=int)
        player_winner_mask[0] = 1
        player_wins = (winners == player_winner_mask).all(1)
        player_wins = np.sum(player_wins, axis=0)

        num_player_draws = num_player_non_loss - player_wins

        return player_wins, num_player_draws

    def random_evaluation(self, iterations=10000):
        deck = Deck(iterations)
        deck.remove_cards(self.player_cards)
        deck.remove_cards(self.table_cards)
        deck.random_shuffle(self.player_count)
        return self.run(deck, iterations)

    def weighted_evaluation(self, hand_probabilities, iterations=10000, debug=False):
        deck = Deck(iterations)
        deck.remove_cards(self.player_cards)
        deck.remove_cards(self.table_cards)
        deck.weighted_shuffle(hand_probabilities)
        return self.run(deck, iterations=iterations, debug=debug)
