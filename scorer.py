import numpy as np

from sklearn.metrics import *
from numba import jit


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_acc(adj_rec, adj_label, threshold=0.):

    labels_all = adj_label.view(-1).long().cpu().numpy()
    preds_all = (adj_rec > threshold).view(-1).long().cpu().numpy()
    accuracy = precision_score(preds_all, labels_all)

    return accuracy


def get_confusion_matrix(adj_rec, adj_label, threshold=0.):
    labels_all = adj_label.view(-1).long().cpu().numpy()
    preds_all = (adj_rec > threshold).view(-1).long().cpu().numpy()

    return confusion_matrix(y_true=labels_all, y_pred=preds_all)


def get_metric(adj_rec, adj_label, threshold=0., beta=1.0):
    labels_all = adj_label.view(-1).long().cpu().numpy()
    preds_all = (adj_rec > threshold).view(-1).long().cpu().numpy()

    accuracy = accuracy_score(labels_all, preds_all)
    precision = precision_score(labels_all, preds_all)
    recall = recall_score(labels_all, preds_all)
    f_beta = fbeta_score(labels_all, preds_all, beta=beta)

    return accuracy, precision, recall, f_beta


def get_aupr(adj_rec, adj_label):
    labels_all = adj_label.view(-1).long().cpu().numpy()
    preds_all = adj_rec.view(-1).long().cpu().numpy()
    precision, recall, thresholds = precision_recall_curve(labels_all, preds_all)

    return auc(recall, precision)


def get_auroc(adj_rec, adj_label):
    labels_all = adj_label.view(-1).cpu().detach().numpy().astype(np.int_)
    preds_all = adj_rec.view(-1).cpu().detach().numpy()

    return roc_auc_score(labels_all, preds_all)


def get_mcc(adj_rec, adj_label, threshold=0.):
    labels_all = adj_label.view(-1).cpu().detach().numpy().astype(np.int_)
    preds_all = preds_all = (adj_rec > threshold).view(-1).long().cpu().numpy()

    return matthews_corrcoef(labels_all, preds_all)
