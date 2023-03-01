import os.path

import numpy.random
from torch.optim import Adam

from loader import ScDataset
from module import *
from scorer import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


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


def train(model, dst, stopper, threshold, feats):
    beta = 0.5

    optimizer = Adam(model.parameters(), CONFIG.LEARNING_RATE, weight_decay=CONFIG.WEIGHT_DECAY)

    train_graph = dst.train_graph.adjacency_matrix().to_dense().to(CONFIG.DEVICE)
    dst.train_graph = dst.train_graph.to(CONFIG.DEVICE)
    weight_tensor, norm = _compute_loss_para(train_graph)

    model.train()
    print('Total Parameters:{}'.format(sum([p.nelement() for p in model.parameters()])))

    for epoch in range(CONFIG.EPOCH):
        logits = model.forward(dst.train_graph, feats)

        loss = norm * func.binary_cross_entropy_with_logits(
            logits.view(-1), train_graph.view(-1), weight=weight_tensor
        )
        kl_divergence = 0.5 / logits.size(0) * (
                1 + 2 * model.log_std - model.mean ** 2 - torch.exp(model.log_std) ** 2
        ).sum(1).mean()
        loss -= beta * kl_divergence

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = get_acc(logits, train_graph, threshold=threshold)
        auroc = get_auroc(logits, train_graph)
        auprc = get_aupr(logits, train_graph)
        if stopper.step(auroc, model):
            break

        iter_nums = epoch + 1
        print(
            "Epoch:", '%03d' % iter_nums, "loss=", "{:.5f}".format(loss.item()),
            "acc=", "{:.5f}".format(acc), "auroc=", "{:.5f}".format(auroc),
            "auprc=", "{:.5f}".format(auprc)
        )


def test(model, dst, turn, file, threshold, feats):
    ground_truth = dst.graph.adjacency_matrix().to_dense().to(CONFIG.DEVICE)
    test_graph = dst.test_graph.adjacency_matrix().to_dense().to(CONFIG.DEVICE)

    dst.train_graph = dst.train_graph.to(CONFIG.DEVICE)
    dst.test_graph = dst.test_graph.to(CONFIG.DEVICE)

    print('test in test graph')
    accuracy, precision, recall, f_beta, au_roc, au_pr, mcc = [], [], [], [], [], [], []
    for _ in range(turn):
        logits = model(dst.train_graph.to(CONFIG.DEVICE), feats)

        # calculate metric
        confused_matrix = get_confusion_matrix(logits, test_graph, threshold=threshold)
        print(confused_matrix)
        t_accuracy, t_precision, t_recall, t_f_beta = get_metric(
            logits, test_graph, threshold=threshold, beta=1
        )
        t_au_roc = get_auroc(logits, test_graph)
        t_au_pr = get_aupr(logits, test_graph)
        t_mcc = get_mcc(logits, test_graph)

        accuracy.append(t_accuracy)
        precision.append(t_precision)
        recall.append(t_recall)
        f_beta.append(t_f_beta)
        au_roc.append(t_au_roc)
        au_pr.append(t_au_pr)
        mcc.append(t_mcc)

    with open(file=file, mode='w', encoding='UTF-8') as f:
        np.set_printoptions(precision=3)
        print('test in test graph, threshold={}'.format(threshold), file=f)
        print(
            '指标:\naccuracy:{:0.6f}(+-{:0.6f}),precision:{:0.6f}(+-{:0.6f}),recall:{:0.6f}(+-{:0.6f}),\n'
            'f_beta:{:0.6f}(+-{:0.6f}),auroc:{:0.6f}(+-{:0.6f}),aupr:{:0.6f}(+-{:0.6f}),mcc:{:0.6f}(+-{:0.6f})'
            .format(
                np.mean(accuracy), np.std(accuracy, ddof=1), np.mean(precision), np.std(precision, ddof=1),
                np.mean(recall), np.std(recall, ddof=1), np.mean(f_beta), np.std(f_beta, ddof=1),
                np.mean(au_roc), np.std(au_roc, ddof=1), np.mean(au_pr), np.std(au_pr, ddof=1),
                np.mean(mcc), np.std(mcc, ddof=1)
            ), end='\n\n', file=f
        )

    print('test in complete graph')
    accuracy, precision, recall, f_beta, au_roc, au_pr, mcc = [], [], [], [], [], [], []
    for _ in range(turn):
        logits = model(dst.train_graph.to(CONFIG.DEVICE), feats)
        confused_matrix = get_confusion_matrix(logits, ground_truth, threshold=threshold)

        print(confused_matrix)
        t_accuracy, t_precision, t_recall, t_f_beta = get_metric(
            logits, ground_truth, threshold=threshold, beta=1
        )
        t_au_roc = get_auroc(logits, ground_truth)
        t_au_pr = get_aupr(logits, ground_truth)
        t_mcc = get_mcc(logits, ground_truth)

        accuracy.append(t_accuracy)
        precision.append(t_precision)
        recall.append(t_recall)
        f_beta.append(t_f_beta)
        au_roc.append(t_au_roc)
        au_pr.append(t_au_pr)
        mcc.append(t_mcc)

    with open(file=file, mode='a', encoding='UTF-8') as f:
        np.set_printoptions(precision=3)
        print('test in complete graph, threshold={}'.format(threshold), file=f)
        print(
            '指标:\naccuracy:{:0.6f}(+-{:0.6f}),precision:{:0.6f}(+-{:0.6f}),recall:{:0.6f}(+-{:0.6f}),\n'
            'f_beta:{:0.6f}(+-{:0.6f}),auroc:{:0.6f}(+-{:0.6f}),aupr:{:0.6f}(+-{:0.6f}),mcc:{:0.6f}(+-{:0.6f})'
            .format(
                np.mean(accuracy), np.std(accuracy, ddof=1), np.mean(precision), np.std(precision, ddof=1),
                np.mean(recall), np.std(recall, ddof=1), np.mean(f_beta), np.std(f_beta, ddof=1),
                np.mean(au_roc), np.std(au_roc, ddof=1), np.mean(au_pr), np.std(au_pr, ddof=1),
                np.mean(mcc), np.std(mcc, ddof=1)
            ), end='\n\n', file=f
        )

    test_edge_true_index = torch.nonzero(test_graph.view(-1) == 1).detach().cpu().numpy()
    all_edge_false_index = torch.nonzero(ground_truth.view(-1) == 0).detach().cpu().numpy()
    all_edge_true_index = torch.nonzero(ground_truth.view(-1) == 1).detach().cpu().numpy()
    net_density = all_edge_true_index.shape[0] / (all_edge_true_index.shape[0] + all_edge_false_index.shape[0])
    ratio = [1, 4, 9, 19]

    for r in ratio:
        accuracy, precision, recall, f_beta, au_roc, au_pr, mcc = [], [], [], [], [], [], []

        print('test in balanced graph, ratio={}'.format(r))
        for _ in range(turn):
            numpy.random.shuffle(all_edge_false_index)
            test_edges_index = numpy.append(
                test_edge_true_index, all_edge_false_index[: int(r * test_edge_true_index.shape[0])]
            )
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
            t_mcc = get_mcc(balanced_logits, balanced_test_graph, threshold=threshold)

            accuracy.append(t_accuracy)
            precision.append(t_precision)
            recall.append(t_recall)
            f_beta.append(t_f_beta)
            au_roc.append(t_au_roc)
            au_pr.append(t_au_pr)
            mcc.append(t_mcc)

        with open(file=file, mode='a', encoding='UTF-8') as f:
            print(
                'test in balanced dataset(density:{},threshold={}, ratio:{})'.format(net_density, threshold, r),
                file=f
            )
            print(
                '指标:\naccuracy:{:0.6f}(+-{:0.6f}),precision:{:0.6f}(+-{:0.6f}),recall:{:0.6f}(+-{:0.6f}),\n'
                'f_beta:{:0.6f}(+-{:0.6f}),auroc:{:0.6f}(+-{:0.6f}),aupr:{:0.6f}(+-{:0.6f}),mcc:{:0.6f}(+-{:0.6f})'
                .format(
                    np.mean(accuracy), np.std(accuracy, ddof=1), np.mean(precision), np.std(precision, ddof=1),
                    np.mean(recall), np.std(recall, ddof=1), np.mean(f_beta), np.std(f_beta, ddof=1),
                    np.mean(au_roc), np.std(au_roc, ddof=1), np.mean(au_pr), np.std(au_pr, ddof=1),
                    np.mean(mcc), np.std(mcc, ddof=1)
                ), end='\n\n', file=f
            )


def main():
    # datsets = ['Non-Specific Dataset', 'Specific Dataset', 'STRING Dataset']
    # cell_type = ['hESC', 'hHEP', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']
    datsets = ['Lofgof Dataset']
    cell_type = ['mESC']
    n_tf = ['TFs500', 'TFs1000']
    # n_tf = ['TFs500']
    # dataset_names = ['Non-Specific', 'Specific', 'STRING']
    dataset_names = ['Lofgof']
    tf = ['500', '1000']
    # tf = ['500']
    exp_file = 'ExpressionData.csv'
    net_file = 'network.csv'
    root_dir = 'data/raw/Benchmark Dataset'
    save_dir = './data/processed'
    threshold = 0.
    turn = 50
    # split_ratios = [
    #     [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4]
    # ]
    split_ratios = [[0.6, 0.4]]

    raw_dirs = [os.path.join(root_dir, d, c, t) for d in datsets for c in cell_type for t in n_tf]
    file_names = ['{}-{}{}_dot_decoder'.format(d, c, t) for d in dataset_names for c in cell_type for t in tf]

    for raw_dir, file_name in zip(raw_dirs, file_names):
        for split_ratio in split_ratios:
            # train
            dst = ScDataset(
                name=file_name, raw_dir=raw_dir, save_dir=save_dir, exp_file=exp_file,
                net_file=net_file, split_ratio=split_ratio
            )
            n_gene, feat = dst.shape
            feats = dst.graph.ndata.pop('x').float().to(CONFIG.DEVICE)
            model = InferNet(
                in_feats=feat, out_feats=feat, hidden_feats=feat, n_gene=n_gene
            ).to(CONFIG.DEVICE)
            model_checkpoint = os.path.join(
                save_dir, file_name, 'model{}{}_dot_decoder.pt'.format(int(split_ratio[0]*10), int(split_ratio[1]*10))
            )
            stopper = EarlyStopping(patience=20, save_path=model_checkpoint)

            train(model=model, dst=dst, stopper=stopper, threshold=threshold, feats=feats)

            # test
            model = InferNet(
                in_feats=feat, out_feats=feat, hidden_feats=feat, n_gene=n_gene
            ).to(CONFIG.DEVICE)
            model.load_state_dict(torch.load(model_checkpoint))
            model.eval()

            file = os.path.join(
                save_dir, file_name, 'metric{}{}_dot_decoder.txt'.format(int(split_ratio[0]*10), int(split_ratio[1]*10))
            )
            test(model=model, dst=dst, turn=turn, file=file, threshold=threshold, feats=feats)


if __name__ == '__main__':
    main()
