from enums.Suit import Suit
from enums.Value import Value


import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix


class CardHeadLog:
    def __init__(self, name: str, suit_loss, value_loss, pred_suit, act_suit, pred_val, act_val, uuids):
        self.name = name
        self.suit_loss = suit_loss
        self.value_loss = value_loss
        self.uuids = uuids
        with torch.no_grad():
            pred_suit_class = self.get_pred_class(pred_suit)
            self.pred_suits = list(pred_suit_class.detach().cpu().numpy())
            self.act_suits = list(act_suit.detach().cpu().numpy())

            pred_val_class = self.get_pred_class(pred_val)
            self.pred_val = list(pred_val_class.detach().cpu().numpy())
            self.act_val = list(act_val.detach().cpu().numpy())

    def get_pred_class(self, ouput):
        softmax = nn.Softmax(dim=1)
        classes = torch.argmax(softmax(ouput), 1)
        return classes

    def get_accuracy(self, pred, act):
        pred = np.array(pred)
        act = np.array(act)
        return (pred == act).sum() / len(pred)

    def create_confusion_matrix(self, y_true, y_pred, classes):
        cf_matrix = confusion_matrix(y_true, y_pred, labels=[
                                     i for i in range(len(classes))])

        denominator = np.sum(cf_matrix, axis=1)[:, None]
        values = np.true_divide(cf_matrix, denominator)
        df_cm = pd.DataFrame(values, [i for i in classes],  columns=[
                             i for i in classes])
        plt.figure(figsize=(12, 7))
        return sn.heatmap(df_cm, annot=True).get_figure()

    def __add__(self, other):
        self.suit_loss += other.suit_loss
        self.value_loss += other.value_loss

        self.pred_suits += other.pred_suits
        self.pred_val += other.pred_val

        self.act_suits += other.act_suits
        self.act_val += other.act_val

        self.uuids += other.uuids

        return self

    def __truediv__(self, value):
        self.suit_loss /= value
        self.value_loss /= value

        return self

    def __str__(self):
        suit_accuracy = self.get_accuracy(self.pred_suits, self.act_suits)
        val_accuracy = self.get_accuracy(self.pred_val, self.act_val)

        return f"{round(self.suit_loss, 3)}({round(suit_accuracy, 3)}%),{round(self.value_loss, 3)}({round(val_accuracy, 3)}%)"

    def get_logs(self):
        suit_accuracy = self.get_accuracy(self.pred_suits, self.act_suits)
        val_accuracy = self.get_accuracy(self.pred_val, self.act_val)
        return ((f"suit_loss_{self.name}", self.suit_loss),
                (f"value_loss_{self.name}", self.value_loss),
                (f"suit_accuracy_{self.name}", suit_accuracy),
                (f"value_accuracy_{self.name}", val_accuracy))

    def get_images(self):
        return ((f"suit_confusion_{self.name}", self.create_confusion_matrix(self.act_suits, self.pred_suits, [str(suit) for suit in Suit])),
                (f"value_confusion_{self.name}", self.create_confusion_matrix(self.act_val, self.pred_val, [str(value) for value in Value])))

    def get_wrong_samples(self):
        import numpy as np
        pred = np.array(self.pred_suits)
        act = np.array(self.act_suits)
        # indicies = np.where(np.any(pred != act))
        uuids = np.array(self.uuids)
        num_print = 40
        # print(pred[0:num_print])
        # print(act[0:num_print])
        # print(uuids[0:num_print])
        print(uuids[pred != act])
        # return uuids[indicies[0:10]]
