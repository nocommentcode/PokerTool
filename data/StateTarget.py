from enums.StateTargetType import StateTargetType


class StateTarget:
    def __init__(self, classification: str, uuid: str):
        self.uuid = uuid

        targets = {}
        try:
            for value, type in zip(classification.split(','), StateTargetType):
                targets[type] = int(value)
        except Exception as e:
            raise Exception(str(e) + uuid)
        self.targets = targets

    def __getitem__(self, target_type):
        return self.targets[target_type]
