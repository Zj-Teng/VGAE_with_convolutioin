import os
import random

import dgl
import numpy as np
import pandas as pd
import torch
from dgl.data import DGLDataset
from dgl.data.utils import save_graphs, load_graphs
from scipy.sparse import coo_matrix

from config import CONFIG


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
    print(gene_frame)
    adj = np.zeros(shape=(n_gene, n_gene), dtype=int)
    net_frame = pd.read_csv(net_file, header=0)
    # print(net_frame)

    for r in net_frame.itertuples():
        adj[gene_frame.gene_ids[r[1]]][gene_frame.gene_ids[r[2]]] = 1
    adj = coo_matrix(adj)

    return features, adj, gene_frame


def _dr_data_parser(file):
    pass


def _dr_network_parser(file):
    pass


def _divide(src, dst, ratio=0.6):
    shuffle_seed = [i for i in range(src.shape[0])]
    random.shuffle(shuffle_seed)

    partition = int(len(src) * ratio)
    train_src, train_dst, test_src, test_dst = [], [], [], []
    for p in range(partition):
        train_src.append(src[p])
        train_dst.append(dst[p])

    for p in range(partition, len(src)):
        test_src.append(src[p])
        test_dst.append(dst[p])

    train_src = np.array(train_src, dtype=int)
    train_dst = np.array(train_dst, dtype=int)
    test_src = np.array(test_src, dtype=int)
    test_dst = np.array(test_dst, dtype=int)

    return train_src, train_dst, test_src, test_dst


class DrDataset(DGLDataset):

    def __init__(
            self, name: str = '', raw_dir: str = None, save_dir: str = None, force_reload: bool = False,
            verbose: bool = False, exp_file: str = None, net_file: str = None
    ):

        super(DrDataset, self).__init__(
            name=name, raw_dir=raw_dir, save_dir=save_dir, force_reload=force_reload, verbose=verbose
        )

    def process(self):

        # parse data from files
        node_features = _data_parser(self.data_file_name)
        print(node_features.shape)
        src_ids, dst_ids = _network_parser(self.edge_file_name, num_gene=self.num_nodes + 1)
        train_src, train_dst, test_stc, test_dst = _divide(src_ids, dst_ids)

        # transform edge and data into tensors
        node_features = torch.from_numpy(node_features).to(torch.float).to(CONFIG.DEVICE)
        src_ids = torch.from_numpy(src_ids).to(torch.int).to(CONFIG.DEVICE)
        dst_ids = torch.from_numpy(dst_ids).to(torch.int).to(CONFIG.DEVICE)
        train_src = torch.from_numpy(train_src).to(torch.int).to(CONFIG.DEVICE)
        train_dst = torch.from_numpy(train_dst).to(torch.int).to(CONFIG.DEVICE)
        test_stc = torch.from_numpy(test_stc).to(torch.int).to(CONFIG.DEVICE)
        test_dst = torch.from_numpy(test_dst).to(torch.int).to(CONFIG.DEVICE)

        # init graphs as a sample
        for feat in node_features:
            graph = dgl.graph((src_ids, dst_ids), num_nodes=self.num_nodes)
            train_graph = dgl.graph((train_src, train_dst), num_nodes=self.num_nodes)
            test_graph = dgl.graph((test_stc, test_dst), num_nodes=self.num_nodes)

            graph.ndata['x'] = feat
            self.graph_sample.append(graph)
            self.train_samples.append(train_graph)
            self.test_samples.append(test_graph)

    def __getitem__(self, idx):

        try:
            return self.graph_sample[idx]
        except IndexError as e:
            print(e)

    def __len__(self):

        return self.num_sample

    def save(self):

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        graph_path = os.path.join(self.save_path, 'samples.bin')
        save_graphs(graph_path, self.graph_sample)

    def load(self):

        if not os.path.exists(self.save_path):
            raise FileNotFoundError('file not found')

        graph_path = os.path.join(self.save_path, 'samples.bin')
        self.graph_sample = load_graphs(graph_path)

    def has_cache(self):
        return os.path.exists(os.path.join(self.save_path, 'sample.bin'))

    @property
    def graph(self):
        return self.graph_sample[0]

    @property
    def train_graph(self):
        return self.train_samples[0]

    @property
    def test_graph(self):
        return self.test_samples[0]


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

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def process(self):
        features, adj, gene_frame = _sc_parser(
            exp_file=self.exp_file,
            net_file=self.net_file
        )

        graph = dgl.graph(adj)
        graph.ndata['x'] = features

    def has_cache(self):
        return os.path.exists(os.path.join(self.save_path, 'sample.bin'))

    def save(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        graph_path = os.path.join(self.save_path, 'samples.bin')
        save_graphs(graph_path, self.graph_sample)

    def load(self):
        if not os.path.exists(self.save_path):
            raise FileNotFoundError('file not found')

        graph_path = os.path.join(self.save_path, 'samples.bin')
        self.graph_sample = load_graphs(graph_path)


if __name__ == '__main__':
    # _sc_parser(
    #     'data/raw/Benchmark Dataset/Lofgof Dataset/mESC/TFs+500/BL--ExpressionData.csv',
    #     'data/raw/Benchmark Dataset/Lofgof Dataset/mESC/TFs+500/BL--network.csv'
    # )
    pass
