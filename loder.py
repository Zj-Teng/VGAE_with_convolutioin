import os

import dgl
import numpy as np
import pandas as pd
import torch
from dgl.data import DGLDataset
from dgl.data.utils import save_graphs, load_graphs
from scipy.sparse import coo_matrix


def _sc_parser(exp_file, net_file):
    exp_frame = pd.read_csv(exp_file, index_col=0, header=0)
    features = exp_frame.to_numpy(dtype=float)
    n_gene = features.shape[0]
    gene_frame = pd.DataFrame(
        {
            'gene_ids': [i for i in range(n_gene)],
            'gene_name': list(exp_frame.index)
        }
    )

    gene_frame.set_index(['gene_name'], inplace=True)
    # print(gene_frame)
    adj = np.zeros(shape=(n_gene, n_gene), dtype=int)
    net_frame = pd.read_csv(net_file, header=0)
    # print(net_frame)

    for r in net_frame.itertuples():
        adj[gene_frame.gene_ids[r[1]]][gene_frame.gene_ids[r[2]]] = 1
    adj = coo_matrix(adj)
    print(adj)

    return features, adj, gene_frame


def _dr_data_parser(file):
    pass


def _dr_network_parser(file):
    pass


def _split(net, ratio):
    print(net.shape)
    num_edges = net.shape[0]
    num_train, num_test, num_valid = [num_edges * r for r in ratio]
    idx = np.array([i for i in range(num_edges)], dtype=int)
    print(idx)
    np.random.shuffle(idx)
    print(idx)

    train_idx = idx[: num_train]
    valid_idx = idx[num_train: num_train + num_valid]
    test_idx = idx[num_train + num_valid:]


    return train_net, valid_net, test_net


class ScDataset(DGLDataset):

    def __init__(
            self, name: str = '', raw_dir: str = None, save_dir: str = None, force_reload: bool = False,
            verbose: bool = False, exp_file: str = None, net_file: str = None
    ):
        super(ScDataset, self).__init__(
            name=name, raw_dir=raw_dir, save_dir=save_dir, force_reload=force_reload, verbose=verbose
        )

        self.exp_file = os.path.join(raw_dir, exp_file)
        self.net_file = os.path.join(raw_dir, net_file)
        self.graph = None

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.graph

    def process(self):
        features, adj, gene_frame = _sc_parser(
            exp_file=self.exp_file,
            net_file=self.net_file
        )

        self.graph = dgl.graph(adj)
        features = torch.from_numpy(features)
        self.graph.ndata['x'] = features

    def has_cache(self):
        return os.path.exists(os.path.join(self.save_path, '{}.bin'.format(self.name)))

    def save(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        graph_path = os.path.join(self.save_path, '{}.bin'.format(self.name))
        save_graphs(graph_path, self.graph_sample)
        print('file is saved in path:{}'.format(graph_path))

    def load(self):
        if not os.path.exists(self.save_path):
            raise FileNotFoundError('file not found')

        graph_path = os.path.join(self.save_path, '{}.bin'.format(self.name))
        self.graph = load_graphs(graph_path)


if __name__ == '__main__':
    features, adj, gene_frame = _sc_parser(
        'data/raw/Benchmark Dataset/Lofgof Dataset/mESC/TFs+500/BL--ExpressionData.csv',
        'data/raw/Benchmark Dataset/Lofgof Dataset/mESC/TFs+500/BL--network.csv'
    )
    train_net, valid_net, test_net = _split(adj, ratio=[0.6, 0.2, 0.2])
    print(train_net)
    print(valid_net)
    print(test_net)
