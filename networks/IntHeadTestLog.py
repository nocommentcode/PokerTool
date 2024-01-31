import torch
from networks.NetworkLog import NetworkLog


class IntHeadLog(NetworkLog):
    def __init__(self, name, loss, pred, act) -> None:
        self.name = name
        self.loss = loss

        with torch.no_grad():
            pred_class = self.get_pred_class(pred)
            self.accuracy = self.get_accuracy(
                pred_class, act).item()
            self.preds = list(pred_class.detach().cpu().numpy())
            self.acts = list(act.detach().cpu().numpy())

    def __add__(self, other):
        self.loss += other.loss
        self.accuracy += other.accuracy
        self.preds += other.preds
        self.acts += other.acts
        return self

    def __truediv__(self, value):
        self.loss /= value
        self.accuracy /= value
        return self

    def __str__(self):
        return f"{round(self.loss, 3)}({round(self.accuracy, 3)}%)"

    def get_logs(self):
        return ((f"{self.name}_loss", self.loss), (f"{self.name}_accuracy", self.accuracy))

    def get_images(self):
        pred_set = set(self.preds)
        return ((f"confusion_{self.name}", self.create_confusion_matrix(self.act_suits, self.pred_suits, list(pred_set))),)
