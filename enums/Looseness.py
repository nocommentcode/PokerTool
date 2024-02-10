from enum import Enum


class Looseness(Enum):
    Loose = "Loose"
    Normal = "Normal"
    Tight = "Tight"

    def get_factor(self):
        loosness = {
            Looseness.Loose: 1.3,
            Looseness.Normal: 1.0,
            Looseness.Tight: 0.7
        }
        return loosness[self]
