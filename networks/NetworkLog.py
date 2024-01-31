import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix


class NetworkLog:
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
