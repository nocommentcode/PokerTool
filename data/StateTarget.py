from enums.StateTargetType import StateTargetType


class StateTarget:
    def __init__(self, classification: str, uuid: str):
        self.uuid = uuid

        targets = {}

        try:
            labels = classification.split(',')
            targets[StateTargetType.DealerPosition] = int(labels[0])
            targets[StateTargetType.NumPlayerCards] = int(labels[1])
            targets[StateTargetType.NumTableCards] = int(labels[2])
            targets[StateTargetType.Opponents] = [int(op) for op in labels[3:]]
        except Exception as e:
            raise Exception(str(e) + uuid)
        self.targets = targets

    def __getitem__(self, target_type):
        return self.targets[target_type]
