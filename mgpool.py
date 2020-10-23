import torch
import torch_sparse

from torch_geometric.nn import graclus
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_batch
from torch_geometric.utils import get_laplacian


def mgpool(x, pos, edge_index, batch, mask=None):
    adj_values = torch.ones(edge_index.shape[1]).cuda()
    cluster = graclus(edge_index)
    cluster, perm = consecutive_cluster(cluster)

    index = torch.stack([cluster, torch.arange(0, x.shape[0]).cuda()], dim=0)
    values = torch.ones(cluster.shape[0], dtype=torch.float).cuda()
    uniq, inv, counts = torch.unique(cluster, return_inverse=True, return_counts=True)
    newsize = uniq.shape[0]

    origsize = x.shape[0]

    new_batch = pool_batch(perm, batch)
    # Compute random walk graph laplacian:
    laplacian_index, laplacian_weights = get_laplacian(edge_index, normalization='rw')
    laplacian_index, laplacian_weights = torch_sparse.coalesce(laplacian_index, laplacian_weights, m=origsize,
                                                               n=origsize)
    index, values = torch_sparse.coalesce(index, values, m=newsize, n=origsize)  # P^T matrix
    new_feat = torch_sparse.spmm(index, values, m=newsize, n=origsize, matrix=x)  # P^T X
    new_pos = torch_sparse.spmm(index, values, m=newsize, n=origsize, matrix=pos)  # P^T POS

    new_adj, new_adj_val = torch_sparse.spspmm(index, values, edge_index, adj_values, m=newsize, k=origsize,
                                               n=origsize, coalesced=True)  # P^T A
    index, values = torch_sparse.transpose(index, values, m=newsize, n=origsize, coalesced=True)  # P
    new_adj, new_adj_val = torch_sparse.spspmm(new_adj, new_adj_val, index, values, m=newsize, k=origsize, n=newsize,
                                               coalesced=True)  # (P^T A) P
    # Precompute QP :
    values = torch.ones(cluster.shape[0], dtype=torch.float).cuda()
    index, values = torch_sparse.spspmm(laplacian_index, laplacian_weights, index, values,
                                        m=origsize, k=origsize, n=newsize, coalesced=True)
    return new_adj, new_feat, new_pos, new_batch, index, values, origsize, newsize


def mgunpool(x, index, values, origsize, newsize):
    # newsize - pooled size, origsize - unpooled size, P comes as nc x n
    index, values = torch_sparse.coalesce(index, values, m=origsize, n=newsize)  # P matrix
    new_feat = torch_sparse.spmm(index, values, m=origsize, n=newsize, matrix=x)  # P^T X

    return new_feat
