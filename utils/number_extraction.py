from typing import List, Tuple
from enums.GameType import GameType
from enums.StackSize import StackSize


def extract_opponent_vals(top_lefts: List[Tuple[int, int]], dims: Tuple[int, int], indexes: List[int]) -> List[int]:
    return [extract_number(tl, dims) for i, tl in enumerate(top_lefts) if i in indexes]


def get_opponent_stack_sizes(game_type: GameType, blinds: int, indexes: List[int]) -> List[StackSize]:
    vals = extract_opponent_vals(
        OPPONENT_STACK_TOP_LEFT[game_type], STACK_DIMS, indexes)
    return [StackSize.from_val(val, blinds) for val in vals]


def get_opponent_ranges(game_type: GameType, indexes: List[int]) -> List[int]:
    return extract_opponent_vals(
        OPPONENT_CALL_RANGE_TOP_LEFT[game_type], CALL_RANGE_DIMS, indexes)


def get_hero_stack_size(blinds: int) -> StackSize:
    val = extract_number(HERO_STACK_TOP_LEFT, STACK_DIMS)
    return StackSize.from_val(val, blinds)
