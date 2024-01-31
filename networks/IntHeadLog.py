from networks.NetworkLog import NetworkLog


import torch


class IntHeadLog(NetworkLog):
    def __init__(self, name, num_classes, loss, pred, act) -> None:
        self.name = name
        self.loss = loss
        self.num_classes = num_classes

        with torch.no_grad():
            pred_class = self.get_pred_class(pred)

            self.preds = list(pred_class.detach().cpu().numpy())
            self.acts = list(act.detach().cpu().numpy())

    def __add__(self, other):
        self.loss += other.loss
        self.preds += other.preds
        self.acts += other.acts
        return self

    def __truediv__(self, value):
        self.loss /= value
        return self

    def __str__(self):
        accuracy = self.get_accuracy(self.preds, self.acts)
        return f"{round(self.loss, 3)}({round(accuracy, 3)}%)"

    def get_logs(self):
        accuracy = self.get_accuracy(self.preds, self.acts)

        return ((f"{self.name}_loss", self.loss), (f"{self.name}_accuracy", accuracy))

    def get_images(self):
        return ((f"confusion_{self.name}", self.create_confusion_matrix(self.acts, self.preds, list([i for i in range(self.num_classes)]))),)
