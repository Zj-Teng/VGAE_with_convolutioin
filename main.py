import os.path

import numpy.random
from torch.optim import Adam

from loader import ScDataset
from module import *
from scorer import *


# TODO: make fp low down(modify function _calc_threshold)

class EarlyStopping:
    def __init__(self, patience=10, save_path=None):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_path = save_path

    def step(self, score, model):
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
        """
        Saves model when validation loss decrease.

        """
        torch.save(model.state_dict(), self.save_path)


def _compute_loss_para(adj):
    pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(CONFIG.DEVICE)

    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm


def _calc_threshold(x, ratio=95):
    logits = x.view(-1).detach().cpu().numpy()
    threshold = np.percentile([lgs for lgs in logits], ratio)
    if threshold < 0.:
        threshold = 0.

    return 0.


def main():
    # init component
    raw_dir = 'data/raw/Benchmark Dataset/STRING Dataset/mHSC-L/TFs1000'
    save_dir = './data/processed'
    file_name = 'STRING-mHSC-L1000'
    model_checkpoint = os.path.join(save_dir, file_name, 'model.pt')
    beta = 0.5
    shape = (692, 200)

    model = InferNet(
        in_feats=200, out_feats=200, hidden_feats=200, shape=shape
    ).to(CONFIG.DEVICE)
    dst = ScDataset(
        name=file_name, raw_dir=raw_dir, save_dir=save_dir,
        exp_file='ExpressionData.csv', net_file='network.csv'
    )
    optimizer = Adam(model.parameters(), CONFIG.LEARNING_RATE, weight_decay=CONFIG.WEIGHT_DECAY)
    print('Total Parameters:{}'.format(sum([p.nelement() for p in model.parameters()])))
    stopper = EarlyStopping(patience=50, save_path=model_checkpoint)

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
        loss -= beta * kl_divergence

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = get_acc(logits, train_graph, threshold=threshold)

        model.eval()
        logits = model(dst.train_graph, feats)
        val_roc = get_auroc(logits, valid_graph)
        val_aupr = get_aupr(logits, valid_graph)
        if stopper.step(val_roc, model):
            break

        iter_nums = epoch + 1
        print(
            "Epoch:", '%03d' % iter_nums, "train_loss=", "{:.5f}".format(loss.item()),
            "train_acc=", "{:.5f}".format(train_acc), "val_au_roc=", "{:.5f}".format(val_roc),
            "val_au_pr=", "{:.5f}".format(val_aupr), "threshold={:.5f}".format(threshold)
        )

    # test
    model = InferNet(
        in_feats=200, out_feats=200, hidden_feats=200, shape=shape
    ).to(CONFIG.DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()

    logits = model(dst.train_graph.to(CONFIG.DEVICE), feats)
    threshold = 0.0

    # calculate metric
    confused_matrix = get_confusion_matrix(logits, test_graph, threshold=threshold)
    print(confused_matrix)
    accuracy, precision, recall, f_beta = get_metric(logits, test_graph, threshold=threshold, beta=1)
    au_roc = get_auroc(logits, test_graph)
    au_pr = get_aupr(logits, test_graph)
    with open(os.path.join(save_dir, file_name, 'metric.txt'), mode='w', encoding='UTF-8') as f:
        print('threshold={}'.format(threshold), file=f)
        print(
            '指标:\naccuracy:{:0.6f},precision:{:0.6f},recall:{:0.6f},\n'
            'f_beta:{:0.6f},auroc:{:0.6f},aupr:{:0.6f}'.format(
                accuracy, precision, recall, f_beta, au_roc, au_pr
            ), end='\n\n', file=f
        )

    print('test in complete graph')
    logits = model(dst.train_graph.to(CONFIG.DEVICE), feats)
    threshold = _calc_threshold(logits)
    confused_matrix = get_confusion_matrix(logits, ground_truth, threshold=threshold)

    print(confused_matrix)
    accuracy, precision, recall, f_beta = get_metric(logits, ground_truth, threshold=threshold, beta=1)
    au_roc = get_auroc(logits, ground_truth)
    au_pr = get_aupr(logits, ground_truth)
    with open(os.path.join(save_dir, file_name, 'metric.txt'), mode='a', encoding='UTF-8') as f:
        print('threshold={}'.format(threshold), file=f)
        print(
            '指标:\naccuracy:{:0.6f},precision:{:0.6f},recall:{:0.6f},\n'
            'f_beta:{:0.6f},auroc:{:0.6f},aupr:{:0.6f}'.format(
                accuracy, precision, recall, f_beta, au_roc, au_pr
            ), end='\n\n', file=f
        )

    print('test in balanced dataset')
    accuracy, precision, recall, f_beta, au_roc, au_pr = [], [], [], [], [], []
    for _ in range(100):
        test_edge_true_index = torch.nonzero(test_graph.view(-1) == 1).detach().cpu().numpy()
        all_edge_false_index = torch.nonzero(ground_truth.view(-1) == 0).detach().cpu().numpy()
        numpy.random.shuffle(all_edge_false_index)
        test_edges_index = numpy.append(test_edge_true_index, all_edge_false_index[: test_edge_true_index.shape[0]])
        test_edges_index = numpy.reshape(test_edges_index, -1)
        balanced_test_graph = torch.take(ground_truth, torch.from_numpy(test_edges_index).to(CONFIG.DEVICE))

        logits = model(dst.train_graph.to(CONFIG.DEVICE), feats)
        balanced_logits = torch.take(logits, torch.from_numpy(test_edges_index).to(CONFIG.DEVICE))
        confused_matrix = get_confusion_matrix(balanced_logits, balanced_test_graph, threshold=threshold)
        print(confused_matrix)
        t_accuracy, t_precision, t_recall, t_f_beta = get_metric(
            balanced_logits, balanced_test_graph, threshold=threshold, beta=1
        )
        t_au_roc = get_auroc(balanced_logits, balanced_test_graph)
        t_au_pr = get_aupr(balanced_logits, balanced_test_graph)

        accuracy.append(t_accuracy)
        precision.append(t_precision)
        recall.append(t_recall)
        f_beta.append(t_f_beta)
        au_roc.append(t_au_roc)
        au_pr.append(t_au_pr)

    accuracy = sum(accuracy) / len(accuracy)
    precision = sum(precision) / len(precision)
    recall = sum(recall) / len(recall)
    f_beta = sum(f_beta) / len(f_beta)
    au_roc = sum(au_roc) / len(au_roc)
    au_pr = sum(au_pr) / len(au_pr)

    with open(os.path.join(save_dir, file_name, 'metric.txt'), mode='a', encoding='UTF-8') as f:
        print('average metric in 100 turn', file=f)
        print(
            '指标:\naccuracy:{:0.6f},precision:{:0.6f},recall:{:0.6f},\n'
            'f_beta:{:0.6f},auroc:{:0.6f},aupr:{:0.6f}'.format(
                accuracy, precision, recall, f_beta, au_roc, au_pr
            ), end='\n\n', file=f
        )


if __name__ == '__main__':
    main()
