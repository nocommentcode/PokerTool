from enum import Enum


class Action(Enum):
    RFI = "rfi"
    RaiseUTG = "raise_utg"
    RaiseHJ = "raise_hj"
    RaiseCO = "raise_co"
    RaiseBTN = "raise_btn"
    RaiseSB = "raise_sb"
    RaiseBB = "raise_bb"

    def __str__(self):
        if self.value == Action.RFI.value:
            return "First to raise"

        if self.value == Action.RaiseUTG.value:
            return "Raised by 3"

        if self.value == Action.RaiseHJ.value:
            return "Raised by 4"

        if self.value == Action.RaiseCO.value:
            return "Raised by 5"

        if self.value == Action.RaiseBTN.value:
            return "Raised by BTN"

        if self.value == Action.RaiseSB.value:
            return "Raised by 1"

        if self.value == Action.RaiseBB.value:
            return "Raised by 2"

    @staticmethod
    def from_string(string):
        actions = {"rfi": Action.RFI,
                   "raise_utg": Action.RaiseUTG,
                   "raise_hj": Action.RaiseHJ,
                   "raise_co": Action.RaiseCO,
                   "raise_btn": Action.RaiseBTN,
                   "raise_sb": Action.RaiseSB,
                   "raise_bb": Action.RaiseBB}
        return actions[string]
