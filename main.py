import torch
import torch.nn.functional as func
from torch.optim import Adam

from config import CONFIG
from loader import ScDataset
from module import InferNet
from scorer import *


# TODO: make fp low down(modify function _calc_threshold)

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


def _calc_threshold(x, ratio=99):
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
        name='mESC', raw_dir='./data/raw/Benchmark Dataset/Lofgof Dataset/mESC/TFs500',
        save_dir='./data/processed', exp_file='ExpressionData.csv', net_file='network.csv'
    )
    optimizer = Adam(model.parameters(), CONFIG.LEARNING_RATE, weight_decay=CONFIG.WEIGHT_DECAY)
    print('Total Parameters:{}'.format(sum([p.nelement() for p in model.parameters()])))

    # generate input
    feats = dst.graph.ndata.pop('x').float().to(CONFIG.DEVICE)
    ground_truth = dst.graph.adjacency_matrix().to_dense().to(CONFIG.DEVICE)
    train_graph = dst.train_graph.adjacency_matrix().to_dense().to(CONFIG.DEVICE)
    valid_graph = dst.valid_graph.adjacency_matrix().to_dense().to(CONFIG.DEVICE)
    test_graph = dst.test_graph.adjacency_matrix().to_dense().to(CONFIG.DEVICE)

    dst.train_graph = dst.train_graph.to(CONFIG.DEVICE)
    dst.valid_graph = dst.valid_graph.to(CONFIG.DEVICE)
    dst.test_graph = dst.test_graph.to(CONFIG.DEVICE)

    # compute loss parameters
    weight_tensor, norm = _compute_loss_para(train_graph)

    # init loader
    # sampler = dgl.dataloading.as_edge_prediction_sampler(
    #     sampler=dgl.dataloading.MultiLayerFullNeighborSampler(10),
    #     negative_sampler=dgl.dataloading.negative_sampler.GlobalUniform(5)
    # )
    # dataloader = dgl.dataloading.DataLoader(
    #     dst.train_graph, sampler=sampler, device='cuda', indices=
    #     batch_size=8, shuffle=True, graph_sampler=sampler
    # )

    for epoch in range(CONFIG.EPOCH):
        model.train()
        logits = model.forward(dst.train_graph, feats)
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

        train_acc = get_acc(logits, train_graph, threshold=threshold)

        model.eval()
        logits = model(dst.train_graph, feats)
        val_roc = get_auroc(logits, valid_graph)
        val_aupr = get_aupr(logits, valid_graph)
        iter_nums = epoch + 1
        print(
            "Epoch:", '%03d' % iter_nums, "train_loss=", "{:.5f}".format(loss.item()),
            "train_acc=", "{:.5f}".format(train_acc),
            "val_au_roc=", "{:.5f}".format(val_roc),
            "val_au_pr=", "{:.5f}".format(val_aupr),
            "threshold={:.5f}".format(threshold)
        )

    # test
    model.eval()
    logits = model(dst.train_graph, feats)
    threshold = _calc_threshold(logits)

    # calculate metric
    confused_matrix = get_confusion_matrix(logits, test_graph, threshold=threshold)
    print(confused_matrix)
    accuracy, precision, recall, f_beta = get_metric(logits, test_graph, threshold=threshold, beta=1)
    au_roc = get_auroc(logits, test_graph)
    au_pr = get_aupr(logits, test_graph)
    print('threshold={}'.format(threshold))
    print(
        '指标:\naccuracy:{:0.6f},precision:{:0.6f},recall:{:0.6f},\nf_beta:{:0.6f},auroc:{:0.6f},aupr:{:0.6f}'.format(
            accuracy, precision, recall, f_beta, au_roc, au_pr
        )
    )

    print('test in complete graph')
    logits = model(dst.graph, feats)
    threshold = _calc_threshold(logits)
    confused_matrix = get_confusion_matrix(logits, test_graph, threshold=threshold)

    print(confused_matrix)
    accuracy, precision, recall, f_beta = get_metric(logits, test_graph, threshold=threshold, beta=1)
    au_roc = get_auroc(logits, test_graph)
    au_pr = get_aupr(logits, test_graph)
    print('threshold={}'.format(threshold))
    print(
        '指标:\naccuracy:{:0.6f},precision:{:0.6f},recall:{:0.6f},\nf_beta:{:0.6f},auroc:{:0.6f},aupr:{:0.6f}'.format(
            accuracy, precision, recall, f_beta, au_roc, au_pr
        )
    )


if __name__ == '__main__':
    main()
