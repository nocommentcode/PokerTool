class IntHeadTrainLog:
    def __init__(self, loss):
        self.loss = loss

    def __add__(self, other):
        self.loss += other.loss
        return self

    def __truediv__(self, value):
        self.loss /= value
        return self

    def __str__(self):
        return f"{round(self.loss, 3)}"

    def get_logs(self):
        return (("train_loss", self.loss),)
