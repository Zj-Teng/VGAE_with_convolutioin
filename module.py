import torch
import torch.nn.functional as func

from dgl.nn.pytorch.conv import SAGEConv, GATv2Conv
from torch import nn
from config import CONFIG


# TODO: construct a new module based on GAT

class InferNet(nn.Module):

    def __init__(
            self, in_feats: int = -1, out_feats: int = -1, hidden_feats: int = -1, n_gene: int = 0
    ):
        super(InferNet, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.hidden_feats = hidden_feats
        self.mean = None
        self.log_std = None

        self.base_encoder = nn.Sequential(
            SAGEConv(in_feats=in_feats, out_feats=hidden_feats, activation=func.relu, aggregator_type='mean'),
            SAGEConv(in_feats=hidden_feats, out_feats=hidden_feats, activation=func.relu, aggregator_type='mean'),
            SAGEConv(in_feats=hidden_feats, out_feats=hidden_feats, activation=func.relu, aggregator_type='mean')
        )

        self.sage_mu = SAGEConv(
            in_feats=hidden_feats, out_feats=out_feats, activation=lambda x: x, aggregator_type='mean'
        )
        self.sage_log_std = SAGEConv(
            in_feats=hidden_feats, out_feats=out_feats, activation=lambda x: x, aggregator_type='mean'
        )

        self.linear = DenseNet(
            in_feats=n_gene, hidden_feats=n_gene, out_feats=n_gene, num_layers=4, dropout=0.2
        )

        self.norm_1 = nn.BatchNorm1d(hidden_feats)
        self.norm_2 = nn.BatchNorm1d(out_feats)

    def encoder(self, graph, feats):
        h = self.base_encoder[0](graph, feats)
        h = self.base_encoder[1](graph, h)
        h = self.base_encoder[2](graph, h)
        h = self.norm_1(h)
        self.mean = self.sage_mu(graph, h)
        self.log_std = self.sage_log_std(graph, h)

        gaussian_noise = torch.randn(h.size(0), self.out_feats).to(CONFIG.DEVICE)
        sampled_z = self.mean + gaussian_noise * torch.exp(self.log_std).to(CONFIG.DEVICE)
        sampled_z = self.norm_2(sampled_z)

        return sampled_z

    def decoder(self, z):

        # adj_rec = _self_conv(z, z)
        adj_rec = self.dot_decoder(z)
        # adj_rec = self.linear(adj_rec)
        # adj_rec = self.causal_inference_decoder(z)

        return adj_rec

    def dot_decoder(self, z):
        adj_rec = torch.matmul(z, torch.transpose(z, 0, 1))

        return adj_rec

    def causal_inference_decoder(self, z):
        adj_rec = torch.matmul(z, torch.transpose(z, 0, 1))
        adj_rec = torch.sigmoid(adj_rec)

        return adj_rec

    def forward(self, g, x):
        z = self.encoder(g, x)
        adj_rec = self.decoder(z)

        return adj_rec


class DenseNet(nn.Module):

    def __init__(self, in_feats, hidden_feats, out_feats, num_layers, dropout):
        super(DenseNet, self).__init__()

        self.in_feats = in_feats
        self.hidden_feats = hidden_feats
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.dropout = dropout

        self.dense = self._init_linear_layer()

    def _init_linear_layer(self):

        layer = nn.ModuleList()

        layer.append(
            nn.Sequential(
                nn.LayerNorm(self.in_feats),
                nn.Dropout(self.dropout),
                nn.Linear(in_features=self.in_feats, out_features=self.hidden_feats),
                nn.ELU()
            )
        )
        for _ in range(self.num_layers - 2):
            unit = nn.Sequential(
                nn.LayerNorm(self.hidden_feats),
                nn.Dropout(self.dropout),
                nn.Linear(in_features=self.hidden_feats, out_features=self.hidden_feats),
                nn.ELU()
            )
            layer.append(unit)
        layer.append(
            nn.Sequential(
                nn.LayerNorm(self.hidden_feats),
                nn.Dropout(self.dropout),
                nn.Linear(in_features=self.hidden_feats, out_features=self.out_feats),
                nn.ELU()
            )
        )

        return layer

    def forward(self, features):

        output = features
        for m in self.dense:
            output = m(output)

        return output


class GATAutoEncoder(nn.Module):
    """
    no use and need to be research.
    """

    def __init__(self, in_feats, hidden_feats, out_feats, num_head):
        super(GATAutoEncoder, self).__init__()

        self.base_encoder = nn.Sequential(
            GATv2Conv(in_feats=in_feats, out_feats=out_feats, num_heads=num_head, feat_drop=0.2, attn_drop=0.2),
            GATv2Conv(in_feats=in_feats, out_feats=out_feats, num_heads=num_head, feat_drop=0.2, attn_drop=0.2),
            GATv2Conv(in_feats=in_feats, out_feats=out_feats, num_heads=num_head, feat_drop=0.2, attn_drop=0.2)
        )
        self.mu = SAGEConv(
            in_feats=hidden_feats, out_feats=out_feats, activation=lambda x: x, aggregator_type='mean'
        )
        self.log_std = SAGEConv(
            in_feats=hidden_feats, out_feats=out_feats, activation=lambda x: x, aggregator_type='mean'
        )

    def encoder(self, x, adj):
        h = self.base_encoder[0](adj, x)
        h = torch.mean(h, dim=0)
        print(h.shape)

    def decoder(self):
        pass

    def forward(self, x, adj):
        pass


def _self_conv(signal, moment):
    signal = torch.fft.fft2(input=signal, norm='ortho')
    moment = torch.fft.fft2(input=moment, norm='ortho')
    adj_rec = torch.matmul(moment, signal.t())
    adj_rec = torch.fft.ifft2(input=adj_rec, norm='ortho').to(torch.float32)

    return adj_rec
