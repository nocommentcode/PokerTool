from termcolor import colored
import numpy as np


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


class PrettyTable:
    def __init__(self, title, header_colour, col_padding):
        self.header_colour = header_colour
        self.col_padding = col_padding
        self.title = title
        self.data_formater = None

    def get_header_formatter(self, underline=False):
        if self.header_colour == "blue":
            func = blue_text

        def formatter(text): return func(text, bold=True, underlined=underline)
        return formatter

    def add_data(self, data, col_names, formatter=None):
        self.data = data
        self.col_names = col_names
        self.data_formater = formatter

    def add_row_names(self, rownames):
        self.rownames = rownames

    def blank(self, count):
        return ' ' * count

    def pad_cell(self, val, width, formatter):
        text_width = len(str(val))
        padding = (width - text_width) // 2
        extra_padding = 1 if ((width - text_width) % 2 == 1) else 0
        return f"{self.blank(padding+extra_padding)}{formatter(val)}{self.blank(padding)}"

    def row_to_string(self, col_widths, row_vals, formatter):
        row = []
        for val, width in zip(row_vals, col_widths):
            row.append(self.pad_cell(val, width, formatter))

        return self.blank(self.col_padding).join(row)

    def str_title(self, col_widths):
        total_width = sum(col_widths) + self.col_padding * \
            (len(self.col_names) - 1)
        title_row = self.pad_cell(
            self.title, total_width, self.get_header_formatter(underline=True))

        return title_row

    def str_header(self, col_widths):
        header_row = [self.blank(col_widths[0])] + [str(name)
                                                    for name in self.col_names]
        header_row = self.row_to_string(
            col_widths, header_row, self.get_header_formatter())

        return header_row

    def str_data(self, col_widths):
        header_formatter = self.get_header_formatter()

        def data_formatter(value):
            if value in self.rownames:
                return header_formatter(str(value))

            if self.data_formater is not None:
                return self.data_formater(value)

            return str(value)

        data_rows = [self.row_to_string(
            col_widths, [name] + vals.tolist(), data_formatter) for name, vals in zip(self.rownames, self.data)]

        return data_rows

    def __str__(self):
        col_widths = self.get_col_widths()

        all_rows = [self.str_title(col_widths),  self.str_header(
            col_widths), *self.str_data(col_widths)]

        return "\n".join(all_rows)

    def get_col_widths(self):
        row_max = max(len(str(rowname)) for rowname in self.rownames)

        col_names = [len(str(col_name)) for col_name in self.col_names]
        _, cols = self.data.shape
        col_vals = [max(len(str(val)) for val in self.data[:, col])
                    for col in range(cols)]
        col_max = [max(name, val) for name, val in zip(col_names, col_vals)]

        return [row_max] + col_max


if __name__ == "__main__":
    table = PrettyTable("Test title", "blue", 3)
    rows = 4
    cols = 3

    table.add_row_names([f"Row {str(i + 1)}"for i in range(rows)])
    col_names = [f"Column {str(i + 1)}"for i in range(cols)]

    data = np.random.random((rows, cols))
    data = data.round(2)

    def formatter(value):
        if value >= 0.7:
            return green_text(str(value))

        if value >= 0.35:
            return yellow_text(str(value))

        return red_text(str(value))

    table.add_data(data, col_names, formatter=formatter)
    print(str(table))
