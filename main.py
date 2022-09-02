import torch
import torch.nn.functional as func
from torch.optim import Adam

from config import CONFIG
from loader import ScDataset
from module import InferNet
from scorer import *


# import scipy


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        r"""Saves model when validation loss decrease."""
        torch.save(model.state_dict(), 'es_checkpoint.pt')


def _compute_loss_para(adj):
    pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(CONFIG.DEVICE)

    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm


def _calc_threshold(x, ratio=98):
    logits = x.view(-1).detach().cpu().numpy()
    threshold = np.percentile([lgs for lgs in logits], ratio)
    if threshold < 0.:
        threshold = 0.

    return threshold


def main():
    # init component
    model = InferNet(
        in_feats=200, out_feats=200, hidden_feats=100, shape=(1120, 200), rand_init=False
    ).to(CONFIG.DEVICE)
    dst = ScDataset(
        name='mESC', raw_dir='./data/raw/Benchmark Dataset/Lofgof Dataset/mESC/TFs+500',
        save_dir='./data/processed', exp_file='BL--ExpressionData.csv', net_file='BL--network.csv'
    )
    optimizer = Adam(model.parameters(), CONFIG.LEARNING_RATE, weight_decay=CONFIG.WEIGHT_DECAY)
    print('Total Parameters:{}'.format(sum([p.nelement() for p in model.parameters()])))

    # init recorder
    train_acc, val_roc, val_ap, test_roc, test_ap = [], [], [], [], []
    logits = None
    threshold = None

    # generate input
    ground_truth = dst.graph.to(CONFIG.DEVICE)
    train_graph = dst.train_graph.to(CONFIG.DEVICE)
    valid_graph = dst.valid_graph.to(CONFIG.DEVICE)
    test_graph = dst.test_graph.to(CONFIG.DEVICE)
    feats = ground_truth.ndata.pop('x').to(CONFIG.DEVICE)

    # compute loss parameters
    weight_tensor, norm = _compute_loss_para(train_graph)

    for epoch in range(CONFIG.EPOCH):
        model.train()
        logits = model.forward(train_graph, feats)
        threshold = _calc_threshold(logits)

        # compute loss
        loss = norm * func.binary_cross_entropy_with_logits(
            logits.view(-1), train_graph.view(-1), weight=weight_tensor
        )
        kl_divergence = 0.5 / logits.size(0) * (
                1 + 2 * model.log_std - model.mean ** 2 - torch.exp(model.log_std) ** 2
        ).sum(1).mean()
        loss -= kl_divergence

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc.append(get_acc(logits, train_graph, threshold=threshold))

        model.eval()
        logits = model(train_graph, feats)
        val_roc.append(get_auroc(valid_graph, logits))
        iter_nums = epoch + 1
        print(
            "Epoch:", '%03d' % iter_nums, "train_loss=", "{:.5f}".format(loss.item()), "train_acc=",
            "{:.5f}".format((sum(train_acc) / iter_nums)), "val_auc=", "{:.5f}".format((sum(val_roc) / iter_nums)),
            "threshold={:.5f}".format(threshold)
        )

    # init
    model.eval()
    accuracy, precision, recall, f_beta, tpr, fpr, aupr, mcc, tnr = [], [], [], [], [], [], [], [], []
    logits = model(train_graph, feats)
    threshold = _calc_threshold(logits, ratio=95)

    # calculate metric
    print(logits)
    confused_matrix = get_confusion_matrix(logits, test_graph, threshold=threshold)
    print(confused_matrix)
    metrix = get_metric(confused_matrix, beta=1)
    accuracy.append(float(metrix.get('accuracy')))
    precision.append(float(metrix.get('precision')))
    recall.append(float(metrix.get('recall')))
    f_beta.append(float(metrix.get('f_beta')))
    tpr.append(float(metrix.get('tpr')))
    fpr.append(float(metrix.get('fpr')))
    tnr.append(float(metrix.get('tnr')))
    mcc.append(float(metrix.get('mcc')))
    print('threshold={}'.format(threshold))

    print(
        '平均指标:\naccuracy:{},precision:{},recall:{},f_beta:{},\ntpr:{},fpr:{},mcc:{},tnr={}'.format(
            sum(accuracy) / len(accuracy), sum(precision) / len(precision),
            sum(recall) / len(recall), sum(f_beta) / len(f_beta),
            sum(tpr) / len(tpr), sum(fpr) / len(fpr),
            sum(mcc) / len(mcc), sum(tnr) / len(tnr)
        )
    )


if __name__ == '__main__':
    main()
