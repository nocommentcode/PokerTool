from termcolor import colored


def make_color_func(color):
    def color_text(text, highlight_color=None, bold=False, underlined=False):
        attributes = []
        if bold:
            attributes.append("bold")
        if underlined:
            attributes.append("underline")

        return colored(text, color, on_color=highlight_color, attrs=attributes)

    return color_text


red_text = make_color_func("red")
green_text = make_color_func("green")
blue_text = make_color_func("blue")
black_text = make_color_func("black")
white_text = make_color_func("white")
yellow_text = make_color_func("yellow")
