import numpy as np

from sklearn.metrics import *


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_acc(adj_rec, adj_label, threshold=0.3):

    labels_all = adj_label.view(-1).long().cpu().numpy()
    preds_all = (adj_rec > threshold).view(-1).long().cpu().numpy()
    accuracy = precision_score(preds_all, labels_all)

    return accuracy


def get_confusion_matrix(adj_rec, adj_label, threshold=0.3):
    """Calculate confused matrix from predict and ground truth.

    :param adj_rec: (tensor) It contains predicted links.
    :param adj_label: (tensor) It contains true links.
    :param threshold: (float) It indicates the threshold of predicting true links
    :return:
        confusion_matrix: list(list())
    """

    labels_all = adj_label.view(-1).long().cpu().numpy()
    preds_all = (adj_rec > threshold).view(-1).long().cpu().numpy()

    return confusion_matrix(y_true=labels_all, y_pred=preds_all)


def get_metric(matrix, beta=1):
    """Calculate metrix from confused matrix to estimate the efficience of module

    :param matrix: (np.array) It indicates confused_matrix.
    :param beta: (float) It indicates which is more important between precision and recall. If beta > 1, the importance
    of recall is more than precision. If beta = 1, the importance of recall is same as precision. If beta > 1, the
    importance of recall is less than precision.
    :return:
        metrix: (dict) It contains kinds of metrix estimating module include accuracy, precision, recall, f_beta, tpr,
        fpr.
    """

    accuracy = (matrix[0, 0] + matrix[1, 1]) / np.sum(matrix)
    precision = matrix[1, 1] / np.sum(matrix[:, 1])
    recall = matrix[1, 1] / np.sum(matrix[1, :])
    f_beta = (1 + (beta**2)) * precision * recall / ((beta**2)*precision + recall)
    tpr = recall
    fpr = matrix[0, 1] / np.sum((matrix[0, :]))
    tnr = matrix[0, 0] / np.sum(matrix[0, :])
    mcc = (matrix[1, 1] * matrix[0, 0] - matrix[0, 1] * matrix[1, 0]) / \
          ((np.sum(matrix[:, 1]) * np.sum(matrix[1, :]) * np.sum(matrix[0, :]) * np.sum(matrix[:, 0])) ** 0.5)

    metrix = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f_beta': f_beta,
        'tpr': tpr,
        'fpr': fpr,
        'tnr': tnr,
        'mcc': mcc
    }

    return metrix


def get_aupr(adj_rec, adj_label):

    labels_all = adj_label.view(-1).long().cpu().numpy()
    preds_all = adj_rec.view(-1).long().cpu().numpy()
    precision, recall, thresholds = precision_recall_curve(labels_all, preds_all)

    return auc(precision, recall)


def get_auroc(adj_rec, adj_label):
    labels_all = adj_label.view(-1).long().cpu().numpy()
    preds_all = adj_rec.view(-1).long().cpu().numpy()

    return roc_auc_score(labels_all, preds_all)
