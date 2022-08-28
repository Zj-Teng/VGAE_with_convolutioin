import os
import re
import torch
import dgl
import numpy as np
import random

from utils.clog import Logger
from utils import config
from dgl.data import DGLDataset
from dgl.data.utils import save_graphs, load_graphs

logger = Logger(name='parser', level=config.LOGGER_LEVEL).get_logger


def _data_parser(file_path):
    r"""parse data_feature from raw data file

    :param file_path: (str) the path of a raw data file
    :return:
        data_frame: (np.array) it contains data parsed from a raw data file
    """

    features = []

    with open(file=file_path, mode='r', encoding='UTF-8', newline='\n') as f:
        line = f.readline()
        feature = []

        while line is not None and line != '':
            line = f.readline()
            contents = line.split('\t')

            if len(line) > 1:
                tmp = []
                # if run in dream3 and dream4, plz use contents[1:]
                for c in contents[1:]:
                    tmp.append(eval(c))
                feature.append(tmp)
            elif len(feature) != 0:
                features.append(feature)
                feature = []

    features = np.array(features, dtype=float)
    features = np.swapaxes(features, 1, 2)
    logger.info('The shape of data_frames is {}'.format(features.shape))

    return features


def _network_parser(file_path=None, num_gene=-1):
    r"""parse the structure of network from a raw data file

    :param num_gene: (int) the number of genes in network
    :param file_path: (str) the path of a raw data file
    :return:
        edges: (np.array) it contains node-pairs. eg: ([out],[in])
    """

    assert not (num_gene == -1), 'num_gene invalid!'
    assert file_path is not None, 'path is None'
    out_nodes, in_nodes = [], []

    with open(file=file_path, mode='r', encoding='UTF-8', newline='\n') as f:
        line = f.readline()

        def num_parser(s):
            match = re.findall(r'\d*', s)
            match = [int(m) for m in match if m != '']

            return match

        while line is not None and line != '':
            positions = num_parser(line)
            if positions[2] == 1:
                # start at zero
                out_nodes.append(positions[0] - 1)
                in_nodes.append(positions[1] - 1)

            line = f.readline()

    edges = np.array([out_nodes, in_nodes], dtype=int)
    logger.info('The shape of edges is {}'.format(edges.shape))

    return edges


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


class Dataset(DGLDataset):

    def __init__(
            self, name: str = '', raw_dir: str = None, save_dir: str = None, force_reload: bool = False,
            verbose: bool = False, data_file_name: str = None, edge_file_name: str = None, num_nodes: int = -1,
            num_sample: int = -1
    ):
        # valid parameters
        assert raw_dir is not None or save_dir is not None, 'check raw_dir or save_dir\n'
        assert num_nodes != -1 and num_sample != -1, 'check num_nodes\n'

        # before call super(), you must initialize parameters
        self.data_file_name = os.path.join(raw_dir, data_file_name)
        self.edge_file_name = os.path.join(raw_dir, edge_file_name)
        self.num_nodes = num_nodes
        self.num_sample = num_sample
        self.graph_sample = []
        self.train_samples = []
        self.test_samples = []

        super(Dataset, self).__init__(
            name=name, raw_dir=raw_dir, save_dir=save_dir, force_reload=force_reload, verbose=verbose
        )

    def process(self):
        r""" process data and the structure of network

        :return:
            None
        """

        # parse data from files
        node_features = _data_parser(self.data_file_name)
        print(node_features.shape)
        src_ids, dst_ids = _network_parser(self.edge_file_name, num_gene=self.num_nodes)
        train_src, train_dst, test_stc, test_dst = _divide(src_ids, dst_ids)

        # transform edge and data into tensors
        node_features = torch.from_numpy(node_features).to(torch.float).to(config.DEVICE)
        src_ids = torch.from_numpy(src_ids).to(torch.int).to(config.DEVICE)
        dst_ids = torch.from_numpy(dst_ids).to(torch.int).to(config.DEVICE)
        train_src = torch.from_numpy(train_src).to(torch.int).to(config.DEVICE)
        train_dst = torch.from_numpy(train_dst).to(torch.int).to(config.DEVICE)
        test_stc = torch.from_numpy(test_stc).to(torch.int).to(config.DEVICE)
        test_dst = torch.from_numpy(test_dst).to(torch.int).to(config.DEVICE)

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
        r""" according to index of dst, return a sample

        :param idx: (int) index of sample
        :return:
            sample: (dgl.graph) a sample of graph
        """

        try:
            return self.graph_sample[idx]
        except IndexError as e:
            print(e)

    def __len__(self):
        r""" return the number of samples

        :return:
            num_samples: (int) it indicates the number of samples
        """

        return self.num_sample

    def save(self):
        r""" save processed files into a file named *.bin

        :return:
            None
        """
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        graph_path = os.path.join(self.save_path, 'samples.bin')
        save_graphs(graph_path, self.graph_sample)

    def load(self):
        r""" load graph from saved files

        :return:
            None
        """

        if not os.path.exists(self.save_path):
            raise FileNotFoundError('file not found')

        graph_path = os.path.join(self.save_path, 'samples.bin')
        self.graph_sample = load_graphs(graph_path)

    def has_cache(self):
        r""" check if there are files processed or not

        :return:
            flag: (bool) it indicates there are file processed or not
        """

        graph_path = os.path.join(self.save_path, 'sample.bin')
        return os.path.exists(graph_path)

    @property
    def graph(self):
        return self.graph_sample[0]

    @property
    def train_graph(self):
        return self.train_samples[0]

    @property
    def test_graph(self):
        return self.test_samples[0]
