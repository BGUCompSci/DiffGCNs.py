import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import to_dense_adj, contains_self_loops, contains_isolated_nodes
from torch_cluster import knn_graph
import torch.nn as nn
from torch_geometric.nn.inits import reset
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch.nn import functional as F
from message_passing2 import MessagePassing2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch.nn.init import kaiming_normal
import os.path as osp
from torch_geometric.nn.pool import graclus, avg_pool_x
import torch_sparse
from mgpool import mgpool

currPath = osp.dirname(osp.realpath(__file__))
epsilon = 1e-20


class mySequential(nn.Sequential):

    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                x, pos, edge_index, batch, k = inputs
                inputs = module(x, pos, edge_index)
                res = inputs
                inputs = (inputs, pos, edge_index, batch, k)
            else:
                res = module(inputs)
        return res


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def MLP(channels, batch_norm=True, relu=False):
    return Seq(*[
        Seq(nn.Linear(channels[i - 1], channels[i]), BN(channels[i]) if batch_norm else Seq(),
            nn.ReLU() if relu else Seq())
        for i in range(1, len(channels))
    ])


def CreateTransposedAuxGraph(edge_index):
    source = edge_index[0, :]
    target = edge_index[1, :]
    new_edge_index = torch.zeros_like(edge_index)
    new_edge_index[0, :] = target
    new_edge_index[1, :] = source
    return new_edge_index


def CreateAuxGraph(edge_index, pos_i, pos_j, original_vertex_features, pos):
    # Build new auxiliary graph:
    num_graph_verts = original_vertex_features.shape[0]
    new_verts = torch.arange(num_graph_verts,
                             num_graph_verts + len(edge_index[0, :]))  # Add edges in G as vertices in aux G
    new_vertex_pos = torch.cat(
        [pos, torch.zeros(len(new_verts), pos.shape[1]).to(device)]
        , dim=0)
    new_vertex_features = torch.cat(
        [original_vertex_features, torch.zeros(len(new_verts), original_vertex_features.shape[1]).to(device)]
        , dim=0)

    # Compute needed components:
    sources = edge_index[0, :]
    targets = edge_index[1, :]
    edge_indices = num_graph_verts + torch.arange(0, len(edge_index[0, :]))
    edge_indices = edge_indices.to(device)

    new_source_edges = torch.stack((sources, edge_indices), dim=0)
    new_target_edges = torch.stack((targets, edge_indices), dim=0)

    new_vertex_pos[num_graph_verts:, :] = (pos_i + pos_j) / 2
    # Build graph from components:
    edge_index_aux = torch.cat([new_target_edges, new_source_edges], dim=1)

    assert (not contains_self_loops(edge_index_aux))
    assert (not contains_isolated_nodes(edge_index_aux))

    return edge_index_aux, new_vertex_features, new_vertex_pos


class EdgeAggr(MessagePassing):

    def __init__(self, in_planes, aggr='add', **kwargs):
        super(EdgeAggr, self).__init__(aggr=aggr, **kwargs)
        # self.bn = torch.nn.BatchNorm1d(num_features=3*in_planes)
        self.aggr = aggr

    def forward(self, x, edge_index, pos):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_i, x_j, pos_i, pos_j, x):
        dist = torch.pow(pos_i - pos_j, 2)
        dist = torch.sum(dist, dim=1) + epsilon
        subs = pos_i - pos_j
        sx = subs[:, 0]
        sy = subs[:, 1]
        sz = subs[:, 2]
        dx = x_j * sx[:, None]
        dy = x_j * sy[:, None]
        dz = x_j * sz[:, None]

        derivs = torch.cat([dx, dy, dz], dim=1)
        return derivs


class VertToEdge(MessagePassing2):

    def __init__(self, aggr='add', **kwargs):
        super(VertToEdge, self).__init__(aggr=aggr, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        a = 10

    def forward(self, x, edge_index, pos):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, pos_i, pos_j, edge_index, x, pos):
        edge_index_aux, new_vertex_features, new_vertex_pos = CreateAuxGraph(edge_index, pos_i, pos_j, x, pos)
        return edge_index_aux, new_vertex_features, new_vertex_pos


class AvgTranspose(MessagePassing):

    def __init__(self, aggr='mean', **kwargs):
        super(AvgTranspose, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr

    def forward(self, x, edge_index):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j


class GraphTranspose(MessagePassing):

    def __init__(self, aggr='add', **kwargs):
        super(GraphTranspose, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr

    def forward(self, x, edge_index, pos):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_j, pos_i, pos_j):
        pos_i = pos_i - pos_j
        feature_size = int(x_j.shape[1] / 3)
        d2x = x_j[:, 0:feature_size] * pos_i[:, 0][:, None]
        d2y = x_j[:, feature_size: 2 * feature_size] * pos_i[:, 1][:, None]
        d2z = x_j[:, 2 * feature_size:] * pos_i[:, 2][:, None]

        pos_i = torch.pow(pos_i, 2)
        pos_i = torch.sum(pos_i, dim=1) + epsilon
        return torch.cat([d2x, d2y, d2z], dim=1)


class DiffGCNLayer(MessagePassing):
    def __init__(self, in_planes, out_planes, relu=True, **kwargs):
        super(DiffGCNLayer, self).__init__()
        self.VtoE = VertToEdge(aggr='mean')
        self.EdgeAggr = EdgeAggr(in_planes=in_planes, aggr='mean')
        self.avgTrans = AvgTranspose(aggr='mean')
        self.GraphTrans = GraphTranspose(aggr='mean')
        self.combine = MLP([3 * in_planes, out_planes], relu=relu)
        # if in_planes != out_planes:
        #    self.shortcut = MLP([in_planes, out_planes], relu=False)
        self.in_planes = in_planes
        self.out_planes = out_planes

    def forward(self, x, pos, edge_index, batch=None):
        # Create aux graph:
        edge_index_aux, new_vertex_features, new_vertex_pos = self.VtoE(x, edge_index, pos)
        edge_index_aux = edge_index_aux.to(device)
        edge_index_aux = edge_index_aux.to(device)
        new_vertex_features = new_vertex_features.to(device)
        new_vertex_pos = new_vertex_pos.to(device)
        # Gradient via local aggregation:
        edge_aggr = self.EdgeAggr(new_vertex_features, edge_index_aux, new_vertex_pos)

        # Create transposed graph (flipped edges):
        transposed_edge_index_aux = CreateTransposedAuxGraph(edge_index_aux)

        # Calculate Gradient term:
        gradAggr = self.avgTrans(edge_aggr, transposed_edge_index_aux)
        gradAggr = gradAggr[0:x.shape[0], :]
        # Calculate Laplacian term:
        laplacianAggr = self.GraphTrans(edge_aggr, transposed_edge_index_aux, new_vertex_pos)
        laplacianAggr = laplacianAggr[0:x.shape[0], :]
        return F.relu(self.combine(gradAggr))
        if self.out_planes == self.in_planes:
            return F.relu(self.combine(torch.cat([x, gradAggr, laplacianAggr], dim=1)))
        else:
            return F.relu(self.combine(torch.cat([x, gradAggr, laplacianAggr], dim=1)))


class DiffGCNBlock(MessagePassing):

    def __init__(self, in_planes, out_planes, k, blocks=3, pool=False, **kwargs):
        super(DiffGCNBlock, self).__init__()
        self.blocks = blocks
        self.k = k

        self.openLayer = DiffGCNLayer(in_planes, out_planes, relu=False)
        self.layers = self._make_layer(self.openLayer, out_planes, self.blocks)
        self.pool = pool

    def _make_layer(self, openLayer, planes, blocks):
        layers = [openLayer]
        for i in range(1, blocks):
            layers.append(DiffGCNLayer(planes, planes, relu=False))

        return mySequential(*layers)

    def forward(self, x, pos, batch=None, edge_index=None):
        if edge_index is None:
            edge_index = knn_graph(x, self.k, batch, loop=False)
            edge_index = edge_index.to(device)

        if self.pool:
            new_adj, new_feat, new_pos, new_batch, index, values, origsize, newsize = mgpool(x, pos, edge_index, batch)
            return self.layers(new_feat, new_pos, new_adj, new_batch, self.k), new_pos, new_batch, (
                index, values, origsize, newsize)
        else:
            new_pos = pos
            new_batch = batch
            new_feat = x
            new_adj = edge_index

        return self.layers(new_feat, new_pos, new_adj, new_batch, self.k), new_pos, new_batch
