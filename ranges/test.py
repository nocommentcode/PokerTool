import numpy as np
iterations = 2
player_count = 3

player_cards = np.array([1, 2])
table_cards = np.array([4, 5, 6])


def remove_cards_from_deck(cards, deck):
    mask = np.isin(deck, cards, invert=True)
    return deck[mask]


def shuffle_deck(deck):
    # all_decks = (iterations x num_cards)
    all_decks = np.tile(deck, reps=(iterations, 1))
    temp_random = np.random.random(all_decks.shape)
    idx = np.argsort(temp_random, axis=-1)
    return all_decks[np.arange(all_decks.shape[0])[:, None], idx]


# 52 cards starting at 5 so that later code will make arrays starting at 2
deck = np.arange(5, 57)
deck = remove_cards_from_deck(player_cards, deck)
deck = remove_cards_from_deck(table_cards, deck)
deck = shuffle_deck(deck)

cards_to_deal = 5 - len(table_cards)
player_hands = np.zeros((iterations, player_count, 7))
for i in range(player_count):

    # player hole cards
    if i == 0:
        player_hands[:, i, :2] = player_cards

    # opponent hole cards
    else:
        start_idx = cards_to_deal + (i * 2)
        end_idx = start_idx + 2
        player_hands[:, i, :2] = deck[:, start_idx: end_idx]

    # table cards already delt
    player_hands[:, i, 2:2+len(table_cards)] = table_cards

    # random remaining table cards
    player_hands[:, i, 2+len(table_cards):] = deck[:, :cards_to_deal]

print(player_hands[:, 0])
print(player_hands[:, 1])
print(player_hands[:, 2])
