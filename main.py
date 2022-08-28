import torch
import dgl
import torch.nn.functional as func
# import scipy

from dgl.dataloading import GraphDataLoader
from utils import config
from utils.clog import Logger
from loder import Dataset
from scorer import *
from torch import Tensor
from torch.optim import Adam
from module import InferNet

logger = Logger(name='main', level=config.LOGGER_LEVEL).get_logger
logger.info(config.DEVICE)


def _mask_test_edges(graph, adj):
    src, dst = graph.edges()
    edges_all = torch.stack([src, dst], dim=0)
    logger.debug('edges_all:{}\n'.format(edges_all))
    edges_all = edges_all.t().cpu().numpy()
    logger.debug('edges_all:{}\n'.format(edges_all))
    num_test = int(np.floor(edges_all.shape[0] * 0.2))
    num_val = int(np.floor(edges_all.shape[0] * 0.1))

    all_edge_idx = list(range(edges_all.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    train_edge_idx = all_edge_idx[(num_val + num_test):]
    test_edges = edges_all[test_edge_idx]
    val_edges = edges_all[val_edge_idx]
    train_edges = np.delete(edges_all, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):

        a = np.array(a)
        rows_close = np.all(np.around(a - b[:, None], tol) == 0, axis=-1)

        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        # train_edges edges_all
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    logger.debug('test_edges_false:\n{}'.format(test_edges_false))
    logger.debug('val_edges_false:\n{}'.format(val_edges_false))
    assert ~ismember(test_edges_false, edges_all), 'false test edges occur in true edges'
    assert ~ismember(val_edges_false, edges_all), 'false valid edges occur in true edges'
    assert ~ismember(val_edges, train_edges), 'valid edges occur in train edges'
    assert ~ismember(test_edges, train_edges), 'test edges occur in train edges'
    assert ~ismember(val_edges, test_edges), 'valid edges occur in test edges'

    # NOTE: these edge lists only contain single direction of edge!
    return train_edge_idx, val_edges, val_edges_false, test_edges, test_edges_false


def _compute_loss_para(adj):
    pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    weight_mask = adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0)).to(config.DEVICE)

    weight_tensor[weight_mask] = pos_weight
    return weight_tensor, norm


def train(model=None, dst=None, optimizer=Adam) -> None:
    """ train parameters of model

    :param model: (nn.Module)
    :param dst: (dgl.graph)
    :param optimizer: pytorch optimizer object
    :return:
    """

    assert model is not None and dst is not None, 'model or dst is None'

    optimizer = optimizer(model.parameters(), config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    loader = GraphDataLoader(dst)

    train_acc, val_roc, val_ap, test_roc, test_ap = [], [], [], [], []
    logits = None

    # generate input
    graph = dst.graph
    adj_orig = dst.graph.adjacency_matrix().to_dense()

    # build test set with xx% positive links
    train_edge_idx, val_edges, val_edges_false, test_edges, test_edges_false = _mask_test_edges(graph, adj_orig)

    # create train graph
    train_edge_idx = torch.tensor(train_edge_idx, dtype=torch.int32).to(config.DEVICE)
    train_graph = dgl.edge_subgraph(graph, train_edge_idx, relabel_nodes=False)
    adj = train_graph.adjacency_matrix().to_dense().to(config.DEVICE)

    # compute loss parameters
    weight_tensor, norm = _compute_loss_para(adj)

    for idx, g in enumerate(loader):
        train_acc.clear()
        val_roc.clear()
        val_ap.clear()

        print('第{}样本'.format(idx))
        # Extract node features
        feats = g.ndata.pop('x')

        threshold = None
        print('Total Parameters:{}'.format(sum([p.nelement() for p in model.parameters()])))

        for epoch in range(config.EPOCH):
            model.train()
            # create training component
            logits = model.forward(g, feats)
            threshold = _calc_threshold(logits)

            # compute loss
            loss = norm * _LossCalculator.binary_cross_entropy(logits.view(-1), adj.view(-1), weight=weight_tensor)
            kl_divergence = 0.5 / logits.size(0) * (
                    1 + 2 * model.log_std - model.mean ** 2 - torch.exp(model.log_std) ** 2
            ).sum(1).mean()
            loss -= kl_divergence
            loss += func.mse_loss(feats, model.nodes_embedding)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_acc.append(get_acc(logits, adj, threshold=threshold))

            val_roc_tmp, val_ap_tmp = get_scores(val_edges, val_edges_false, logits)
            val_roc.append(val_roc_tmp)
            val_ap.append(val_ap_tmp)
            iter_nums = epoch + 1
            print(
                "Epoch:", '%04d' % iter_nums, "train_loss=", "{:.5f}".format(loss.item()), "train_acc=",
                "{:.5f}".format((sum(train_acc) / iter_nums)), "val_auc=", "{:.5f}".format((sum(val_roc) / iter_nums)),
                "val_ap=", "{:.5f}".format((sum(val_ap) / iter_nums)), "threshold={:.5f}".format(threshold)
            )

        model.eval()
        print(get_confusion_matrix(logits, adj_orig, threshold=threshold))
        test_roc_tmp, test_ap_tmp = get_scores(test_edges, test_edges_false, logits)
        test_roc.append(test_roc_tmp)
        test_ap.append(test_ap_tmp)

    print('test_ap:', test_ap)
    print('test_roc:', test_roc)
    print(
        "End of training!", "test_auc=", "{:.5f}".format(sum(test_roc) / len(test_roc)),
        "test_ap=", "{:.5f}".format(sum(test_ap) / len(test_ap))
    )


def test(model=None, dst=None):
    """

    :param model:
    :param dst:
    :return:
    """

    assert model is not None and dst is not None, 'model or dst is None'
    model.eval()

    loader = GraphDataLoader(dst)
    accuracy, precision, recall, f_beta, tpr, fpr, aupr, mcc, tnr = [], [], [], [], [], [], [], [], []

    # generate ground truth
    adj_orig = dst.graph.adjacency_matrix().to_dense()

    for idx, g in enumerate(loader):
        # Extract node features
        feats = g.ndata.pop('x')

        logits = model(dst.graph, feats)
        threshold = _calc_threshold(logits, ratio=90)

        print(logits)
        confused_matrix = get_confusion_matrix(logits, adj_orig, threshold=threshold)
        print(confused_matrix)
        metrix = get_metrix(confused_matrix, beta=1)
        accuracy.append(float(metrix.get('accuracy')))
        precision.append(float(metrix.get('precision')))
        recall.append(float(metrix.get('recall')))
        f_beta.append(float(metrix.get('f_beta')))
        tpr.append(float(metrix.get('tpr')))
        fpr.append(float(metrix.get('fpr')))
        tnr.append(float(metrix.get('tnr')))
        mcc.append(float(metrix.get('mcc')))
        # aupr.append(get_aupr(logits, adj_orig, threshold=threshold))
        print('第{}样本指标:\n{}\n'.format(idx, metrix))
        print('threshold={}'.format(threshold))

    print(
        '平均指标:\naccuracy:{},precision:{},recall:{},f_beta:{},\ntpr:{},fpr:{},mcc:{},tnr={}'.format(
            sum(accuracy) / len(accuracy), sum(precision) / len(precision),
            sum(recall) / len(recall), sum(f_beta) / len(f_beta),
            sum(tpr) / len(tpr), sum(fpr) / len(fpr),
            sum(mcc) / len(mcc), sum(tnr) / len(tnr)
        )
    )


class _LossCalculator(object):

    @staticmethod
    def binary_cross_entropy(input: Tensor, target: Tensor, weight: Tensor = None):
        return func.binary_cross_entropy_with_logits(input, target, weight=weight)

    @staticmethod
    def mse_loss(input: Tensor, target: Tensor):
        return func.mse_loss(input, target)


def _calc_threshold(x, ratio=95):
    r""" Figure out threshold from positive number.

    :param x: (Tensor) Adjacency matrix
    :return:
    """
    logits = x.view(-1).detach().cpu().numpy()
    threshold = np.percentile([lgs for lgs in logits], ratio)

    return threshold


if __name__ == '__main__':
    base_model = InferNet(
        in_feats=21, out_feats=21, hidden_feats=42, shape=(100, 21), rand_init=False
    ).to(config.DEVICE)
    dataset1 = Dataset(
        name='dr4_Insilico_1', raw_dir='./data/raw', save_dir='./data/processed',
        data_file_name='D4_insilico_size100_1_timeseries.tsv',
        edge_file_name='DREAM4_GoldStandard_InSilico_Size100_1.tsv',
        num_nodes=100, num_sample=10
    )
    dataset2 = Dataset(
        name='dr4_Insilico_2', raw_dir='./data/raw', save_dir='./data/processed',
        data_file_name='D4_insilico_size100_2_timeseries.tsv',
        edge_file_name='DREAM4_GoldStandard_InSilico_Size100_2.tsv',
        num_nodes=100, num_sample=10
    )

    train(base_model, dataset1)
    test(base_model, dataset2)
