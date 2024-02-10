from enum import Enum


class Action(Enum):
    RFI = "rfi"
    RaiseSB = "raise_sb"
    RaiseBB = "raise_bb"
    RaiseUTG = "raise_utg"
    RaiseHJ = "raise_hj"
    RaiseCO = "raise_co"
    RaiseBTN = "raise_btn"

    def __str__(self):
        if self.value == Action.RFI.value:
            return "First to raise"

        if self.value == Action.RaiseUTG.value:
            return "Raised by UTG"

        if self.value == Action.RaiseHJ.value:
            return "Raised by HJ"

        if self.value == Action.RaiseCO.value:
            return "Raised by CO"

        if self.value == Action.RaiseBTN.value:
            return "Raised by BTN"

        if self.value == Action.RaiseSB.value:
            return "Raised by SB"

        if self.value == Action.RaiseBB.value:
            return "Raised by BB"

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
