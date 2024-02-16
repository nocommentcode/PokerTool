from enum import Enum
from utils.printing import green_text, red_text, yellow_text


class StackSize(Enum):
    Ten = 10
    Twelve = 12
    Fifteen = 15
    Twenty = 20
    TwentyFive = 25
    Thirty = 30
    Fourty = 40
    Sixty = 60
    Eighty = 80

    @staticmethod
    def from_val(stack_size, blinds):
        big_blinds = stack_size // blinds
        for stack in sorted(StackSize, key=lambda stack: stack.value, reverse=True):
            if big_blinds >= stack.value:
                return stack

        return StackSize.Ten

    def pretty_str(self):
        all_stacks = list(sorted(StackSize, key=lambda stack: stack.value))
        my_position = all_stacks.index(self)
        stack_strength = my_position // 3
        stack_colours = [red_text, yellow_text, green_text]
        return stack_colours[stack_strength](f"{self.value}BB", bold=True)
