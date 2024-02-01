from enums.StateTargetType import StateTargetType


class StateTarget:
    def __init__(self, classification: str, uuid: str):
        self.uuid = uuid

        targets = {}
        for value, type in zip(classification.split(','), StateTargetType):
            targets[type] = int(value)

        self.targets = targets

    def __getitem__(self, target_type):
        return self.targets[target_type]
