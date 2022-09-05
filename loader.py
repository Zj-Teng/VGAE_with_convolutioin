import os

import dgl
import networkx as nx
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

    return features, adj, gene_frame


def _dr_data_parser(file):
    pass


def _dr_network_parser(file):
    pass


def _split(exp, net, ratio=None):
    if ratio is None:
        ratio = [0.6, 0.2, 0.2]

    g = nx.from_numpy_array(net, create_using=nx.DiGraph)
    node_list = g.nodes
    edge_list = nx.to_edgelist(g)
    edge_list = list(edge_list)
    num_edges = len(edge_list)

    num_train, num_test, num_valid = [int(num_edges * r) for r in ratio]
    idx = np.array([i for i in range(num_edges)], dtype=int)
    np.random.shuffle(idx)
    train_idx = idx[: num_train]
    valid_idx = idx[num_train: (num_train + num_valid)]
    test_idx = idx[(num_train + num_valid):]

    train_edges = [edge_list[i] for i in train_idx]
    valid_edges = [edge_list[i] for i in valid_idx]
    test_edges = [edge_list[i] for i in test_idx]

    def generator(exp, edges, nodes):
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        graph = dgl.from_networkx(graph)
        exp = exp[list(graph.nodes())]
        graph.ndata['x'] = torch.from_numpy(exp)

        return graph

    train_graph = generator(exp, train_edges, node_list)
    valid_graph = generator(exp, valid_edges, node_list)
    test_graph = generator(exp, test_edges, node_list)

    return train_graph, valid_graph, test_graph


class ScDataset(DGLDataset):

    def __init__(
            self, name: str = '', raw_dir: str = None, save_dir: str = None, force_reload: bool = False,
            verbose: bool = True, exp_file: str = None, net_file: str = None
    ):
        self.exp_file = os.path.join(raw_dir, exp_file)
        self.net_file = os.path.join(raw_dir, net_file)
        self.graph = None
        self.train_graph = None
        self.valid_graph = None
        self.test_graph = None
        self.all_graph = list()

        super(ScDataset, self).__init__(
            name=name, raw_dir=raw_dir, save_dir=save_dir, force_reload=force_reload, verbose=verbose
        )

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.graph

    def process(self):
        features, adj, gene_frame = _sc_parser(
            exp_file=self.exp_file,
            net_file=self.net_file
        )

        self.train_graph, self.valid_graph, self.test_graph = _split(features, adj, ratio=[0.6, 0.2, 0.2])

        adj = coo_matrix(adj, dtype=int)
        self.graph = dgl.from_scipy(adj)
        self.graph.ndata['x'] = torch.from_numpy(features)

        self.all_graph.append(self.graph)
        self.all_graph.append(self.train_graph)
        self.all_graph.append(self.valid_graph)
        self.all_graph.append(self.test_graph)

    def has_cache(self):
        print(os.path.join(self.save_path, '{}.bin'.format(self.name)))
        return os.path.exists(os.path.join(self.save_path, '{}.bin'.format(self.name)))

    def save(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        path = os.path.join(self.save_path, '{}.bin'.format(self.name))
        save_graphs(path, self.all_graph)

    def load(self):
        path = os.path.join(self.save_path, '{}.bin'.format(self.name))
        if not os.path.exists(path):
            raise FileNotFoundError('file not found')

        self.all_graph = load_graphs(path)
        self.graph, self.train_graph, self.valid_graph, self.test_graph = self.all_graph[0]


if __name__ == '__main__':
    dst = ScDataset(
        name='mESC', raw_dir='./data/raw/Benchmark Dataset/Lofgof Dataset/mESC/TFs500',
        save_dir='./data/processed', exp_file='ExpressionData.csv', net_file='network.csv'
    )


