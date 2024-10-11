import scipy.sparse as sp
import sklearn
from sklearn.impute import SimpleImputer
import torch
import networkx as nx
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.functional import normalize
from sklearn.cluster import KMeans
import community as community_louvain
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import scanpy as sc
import h5py
import faiss
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import issparse, csr_matrix
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances
from typing import Optional
EPS = 1e-15


# def normalize(adata, highly_genes=3000, normalize_input=True):
#     print("start select HVGs")
#     adata = np.delete(adata, np.mean(adata, axis=0) < 0.04, axis=1)
#     adata1 = np.int64(adata > 0)
#     adata = np.delete(adata, np.sum(adata1, axis=0) < 300, axis=1)
#     adata = np.delete(adata, np.std(adata, axis=0) < 0.1, axis=1)
#     top_k_idx = np.std(adata, axis=0).argsort()[::-1][0:highly_genes]
#     top_k_idx = top_k_idx[::-1]
#     adata = adata[:, top_k_idx]
#     if normalize_input:
#         adata = adata / np.sum(adata, axis=1).reshape(-1, 1) * 10000
#     return sc.AnnData(adata)
def aug_construct_graph(adata,
        alpha: float = 1,
        n_neighbors: int = 10,
        n_pcs: int = 15,
        use_highly_variable: Optional[bool] = True,
        normalize_total: bool = True,  # 使用总数归一化
        # n_components: int = 15,
):
    print('aug_start')
    #MOB这两行注释掉
    #sc.pp.normalize_total(adata) if normalize_total else None
    #sc.pp.log1p(adata)  # log(1+x)对偏度比较大的数据用log1p函数进行转化，使其更加服从高斯分布。
    adata.layers['log1p-ori'] = adata.X
    # 得到一个包含高变异基因名称的列表
    hvg = list(adata.var['highly_variable'][adata.var['highly_variable'].values].index)
    exmatrix_ori = adata.to_df(layer='log1p-ori')[hvg].to_numpy()
    pca_ori = PCA(n_components=1000)
    pca_ori.fit(exmatrix_ori)
    exmatrix_ori = pca_ori.transform(exmatrix_ori)

    sc.pp.pca(adata, n_comps=n_pcs, use_highly_variable=use_highly_variable)

    coord = adata.obsm['spatial']  # 4727*2 (x,y)
    print(coord)
    neigh = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(coord)
    nbrs = neigh.kneighbors_graph(coord)  # 4727*4727 邻接矩阵？
    dists = np.exp(2 - cosine_distances(adata.obsm['X_pca'])) - 1
    conns = nbrs.T.toarray() * dists
    print('aug_start2')
    #代码使用 issparse() 函数检查 adata.X 是否为稀疏矩阵格式。如果 adata.X 是稀疏矩阵，则通过调用 adata.X.toarray() 将其转换为稠密数组形式
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    X_rec = alpha * np.matmul(conns / np.sum(conns, axis=0, keepdims=True), X) + X

    stg = conns / np.sum(conns, axis=0, keepdims=True)
    print('aug_start3')
    adata.X = csr_matrix(X_rec)
    ##############加权的图
    adata.layers['log1p-aug'] = adata.X

    exmatrix_aug = adata.to_df(layer='log1p-aug')[hvg].to_numpy()
    pca_aug = PCA(n_components=1000)
    pca_aug.fit(exmatrix_aug)
    exmatrix_aug = pca_ori.transform(exmatrix_aug)

    del adata.obsm['X_pca']
    # '''记录重构参数'''
    # adata.uns['spatial_reconstruction'] = {}
    # rec_dict = adata.uns['spatial_reconstruction']
    # rec_dict['params'] = {}
    # rec_dict['params']['alpha'] = alpha
    # rec_dict['params']['n_neighbors'] = n_neighbors
    # rec_dict['params']['n_pcs'] = n_pcs
    # rec_dict['params']['use_highly_variable'] = use_highly_variable
    # rec_dict['params']['normalize_total'] = normalize_total
    # #返回四个变量adata，exmatrix_ori, exmatrix_aug, stg
    print('aug_end')
    return adata, exmatrix_ori, exmatrix_aug, stg
def neraest_labels(instance, labels):
    instance = instance.clone().detach()#torch.tensor(instance).float()
    instance = instance.cpu().numpy()
    labels = labels.clone().detach()#torch.tensor(labels).float()
    labels = labels.cpu().numpy()
    # 创建 Faiss 索引
    index = faiss.IndexFlatL2(instance.shape[1])  # 使用 FlatL2 索引
    # 将instance的数据添加到索引中
    index.add(instance)
    # 搜索最近邻
    k = 6
    distances, indices = index.search(instance, k)
    # print('indices', indices.flatten())
    # 获取最近邻样本的标签
    topK_labels = labels[indices]
    topK_labels =torch.from_numpy(topK_labels)
    topK_labels = torch.squeeze(topK_labels, dim=1)
    return torch.FloatTensor(topK_labels)
def regularization_loss(emb, graph_nei, graph_neg):
    mat = torch.sigmoid(cosine_similarity(emb))  # .cpu()
    # mat = pd.DataFrame(mat.cpu().detach().numpy()).values

    # graph_neg = torch.ones(graph_nei.shape) - graph_nei

    neigh_loss = torch.mul(graph_nei, torch.log(mat)).mean()
    neg_loss = torch.mul(graph_neg, torch.log(1 - mat)).mean()
    pair_loss = -(neigh_loss + neg_loss) / 2
    return pair_loss

def mask_correlated_samples(N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask


def forword_feature(h_i, h_j):
    feature_size, _ = h_i.shape
    N = 2 * feature_size
    h = torch.cat((h_i, h_j), dim=0)
    temperature_f = 0.5
    sim = torch.matmul(h, h.T) / temperature_f
    sim_i_j = torch.diag(sim, feature_size)
    sim_j_i = torch.diag(sim, -feature_size)

    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    mask = mask_correlated_samples(N)
    negative_samples = sim[mask].reshape(N, -1)

    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_contrast = criterion(logits, labels)
    loss_contrast /= N

    return 0.1* loss_contrast
def forward_pui_label(ologits, plogits):

        print(ologits.shape,plogits.shape)
        similarity = torch.mm(normalize(ologits.t(), p=2, dim=1), F.normalize(plogits, p=2, dim=0))
        criterion=nn.CrossEntropyLoss(reduction="sum")
        loss_ce = criterion(similarity, torch.arange(similarity.size(0)).cuda())
        # balance regularisation
        o = ologits.sum(0).view(-1)
        o /= o.sum()
        loss_ne = math.log(o.size(0)) + (o * o.log()).sum()
        loss = 0.05 * loss_ce + 0.05 * loss_ne
        return loss
def target_distribution(q):
    p = q ** 2 / torch.sum(q, dim=0)
    p = p / torch.sum(p, dim=1, keepdim=True)
    return p
def target_l2(q):
    return ((q ** 2).t() / (q ** 2).sum(1)).t()
def forword_debiased_instance(h, h_i, y_pred,sample_size):

        sample_size = sample_size
        temperature = 0.5
        y_sam = torch.LongTensor(y_pred)
        neg_size = 128
        class_sam = []
        for m in range(np.max(y_pred) + 1):
            class_del = torch.ones(int(sample_size), dtype=bool)
            class_del[np.where(y_sam.cpu() == m)] = 0
            class_neg = torch.arange(sample_size).masked_select(class_del)
            neg_sam_id = random.sample(range(0, class_neg.shape[0]), int(neg_size))
            class_sam.append(class_neg[neg_sam_id])

        out = h
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        neg_samp = torch.zeros(neg.shape[0], int(neg_size))
        for n in range(np.max(y_pred) + 1):
            neg_samp[np.where(y_sam.cpu() == n)] = neg.cpu().index_select(1, class_sam[n])[np.where(y_sam.cpu() == n)]
        neg_samp = neg_samp.cuda()
        Ng = neg_samp.sum(dim=-1)


        out = h
        pos = torch.exp(torch.mm(out, out.t().contiguous()))
        pos = torch.diag(torch.exp(torch.mm(out, h_i.t().contiguous())))
        loss = (- torch.log(pos / (Ng))).mean()#pos +
        return 0.1 * loss
def cosine_similarity(emb):
    mat = torch.matmul(emb, emb.T)
    norm = torch.norm(emb, p=2, dim=1).reshape((emb.shape[0], 1))
    mat = torch.div(mat, torch.matmul(norm, norm.T))
    if torch.any(torch.isnan(mat)):
        mat = _nan2zero(mat)
    mat = mat - torch.diag_embed(torch.diag(mat))
    return mat


def consistency_loss(emb1, emb2 ,emb3):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb3 = emb3 - torch.mean(emb3, dim=0, keepdim=True)

    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    emb3 = torch.nn.functional.normalize(emb3, p=2, dim=1)

    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cov3 = torch.matmul(emb3, emb3.t())

    # 计算三个嵌入之间的相关性或协方差，然后根据这些相关性或协方差来调整损失函数
    loss = torch.mean((cov1 - cov2) ** 2) + torch.mean((cov1 - cov3) ** 2) + torch.mean((cov2 - cov3) ** 2)

    return loss


def spatial_construct_graph1(adata, radius=150):

    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']
    A=np.zeros((coor.shape[0],coor.shape[0]))

    # print("coor:", coor)
    nbrs = sklearn.neighbors.NearestNeighbors(radius=radius).fit(coor)
    distances, indices = nbrs.radius_neighbors(coor, return_distance=True)

    for it in range(indices.shape[0]):
        A[[it] * indices[it].shape[0], indices[it]]=1
    print("空间图")
    print('The graph contains %d edges, %d cells.' % (sum(sum(A)), adata.n_obs))
    print('%.4f neighbors per cell on average.' % (sum(sum(A)) / adata.n_obs))

    graph_nei = torch.from_numpy(A)

    graph_neg = torch.ones(coor.shape[0],coor.shape[0]) - graph_nei
    print('1')
    sadj = sp.coo_matrix(A, dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    print('空间图结束')
    # nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    # nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    return sadj, graph_nei, graph_neg#, nsadj

def spatial_construct_graph(positions, k=15):
    print("start spatial construct graph")
    A = euclidean_distances(positions)
    tmp = 0
    mink = 2
    for t in range(100, 1000, 100):
        A1 = np.where(A > t, 0, 1)
        if mink < np.min(np.sum(A1, 1)) and k < np.max(np.sum(A1, 1)):
            tmp = t
            break
    for t in range(tmp - 100, 1000, 10):
        A1 = np.where(A > t, 0, 1)
        if mink < np.min(np.sum(A1, 1)) and k < np.max(np.sum(A1, 1)):
            tmp = t
            break
    for t in range(tmp - 10, 1000, 5):
        A1 = np.where(A > t, 0, 1)
        if mink < np.min(np.sum(A1, 1)) and k < np.max(np.sum(A1, 1)):
            A = A1
            break
    row, col = np.diag_indices_from(A)
    A[row, col] = 0
    # index = np.argwhere(A > 0)
    # np.savetxt('./result/edge.csv', index, delimiter=',')

    graph_nei = torch.from_numpy(A)
    # print(type(graph_nei),graph_nei)
    graph_neg = torch.ones(positions.shape[0], positions.shape[0]) - graph_nei

    sadj = sp.coo_matrix(A, dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    # nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    # nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    print('spatial over')
    return sadj, graph_nei, graph_neg#, nsadj




def features_construct_graph1(features, k=15, pca=None, mode="connectivity", metric="cosine"):
    print(type(features))
    print('1')
    from sklearn.metrics import pairwise_distances
    # data,
    # n_components = 50,
    # gene_dist_type = "cosine",
    # ):
    pca = PCA(n_components=50)
    # if isinstance(features, np.ndarray):
    #     data_pca = pca.fit_transform(features)
    # elif isinstance(features, csr_matrix):
    #     data = features.toarray()
    data_pca = pca.fit_transform(features.toarray())
    gene_correlation = 1 - pairwise_distances(data_pca, metric="cosine")
    return gene_correlation


    print("start features construct graph")
    print('2')
    if pca is not None:
        features = dopca(features, dim=pca).reshape(-1, 1)
    # print("k: ", k)
    # print("features_construct_graph features", features.shape)
    A = kneighbors_graph(features, k + 1, mode=mode, metric=metric, include_self=True)
    A = A.toarray()
    row, col = np.diag_indices_from(A)
    A[row, col] = 0
    # index = np.argwhere(A > 0)
    # np.savetxt('./result/fadj.csv', index, delimiter=',')
    fadj = sp.coo_matrix(A, dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    # nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    # nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    return fadj#, nfadj



def features_construct_graph(features, k=15, pca=None, mode="connectivity", metric="cosine"):
    print("start features construct graph")
    print(type(features))
    imputer = SimpleImputer(strategy='mean')  # 使用均值填充 NaN
    features = imputer.fit_transform(features)
    if pca is not None:
        features = dopca(features, dim=pca).reshape(-1, 1)
    A = kneighbors_graph(features, k + 1, mode=mode, metric=metric, include_self=True)
    A = A.toarray()
    row, col = np.diag_indices_from(A)
    A[row, col] = 0
    fadj = sp.coo_matrix(A, dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    return fadj


# def load_data(config):
#     print("load data:")
#
#     labels = pd.read_table(config.label_path, sep='\t')
#     labels = labels["layer_guess_reordered"].copy()
#
#     NA_index = np.where(labels.isnull())
#     labels = labels.drop(labels.index[NA_index])
#
#     adata = h5py.File(config.feature_path, 'r')
#     print("path: ",config.feature_path)
#     data = np.array(adata['matrix']["data"])
#     indices = np.array(adata['matrix']["indices"])
#     indptr = np.array(adata['matrix']["indptr"])
#     shape = np.array(adata['matrix']["shape"])
#     res = csr_matrix((data, indices, indptr), shape=[shape[1], shape[0]]).toarray()
#     print("data shape: ", res.shape)
#     res = np.delete(res, NA_index, axis=0)
#     print("The data shape after delete the spots with label nan: ", res.shape)
#
#     # adata = sc.AnnData(res)
#     adata = normalize(res, highly_genes=config.fdim)
#     # print("features: ", adata.X.shape)
#     features = sp.csr_matrix(adata.X, dtype=np.float32)
#     features = torch.FloatTensor(np.array(features.todense()))
#     fadj, nfadj = features_construct_graph(features, k=config.k)
#
#     positions = pd.read_csv(config.positions_path, sep=',')
#     # print("positions shape: ", positions.shape)
#     index_labels = labels.index
#     index_positions = positions.iloc[:, 0]
#     dict = []
#     for i in range(len(index_labels)):
#         index = index_positions[index_positions == index_labels[i]].index[0]
#         dict.append(positions.iloc[index, [4, 5]])
#     positions = np.array(dict, dtype=float)
#     # print("after positions shape: ", positions.shape)
#     # np.savetxt('./result/positions.csv', positions, delimiter=',')
#     sadj, nsadj, graph_nei, graph_neg = spatial_construct_graph(positions, k=config.k1)
#     # print("index_labels: ", index_labels)
#     # np.savetxt('./result/label.csv', ground, delimiter=',')
#
#
#     sadj = torch.LongTensor(sadj.todense())
#     fadj = torch.LongTensor(fadj.todense())
#
#
#     print("done")
#
#
#
#     return features, labels, nsadj, nfadj, graph_nei, graph_neg





def get_adj(data, pca=None, k=25, mode="connectivity", metric="cosine"):
    if pca is not None:
        data = dopca(data, dim=pca)
        data = data.reshape(-1, 1)
    A = kneighbors_graph(data, k, mode=mode, metric=metric, include_self=True)
    adj = A.toarray()
    adj_n = norm_adj(adj)
    # S = cosine_similarity(data)
    return adj, adj_n  # , S


def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output


def dopca(data, dim=50):
    return PCA(n_components=dim).fit_transform(data)


def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


class louvain:
    def __init__(self, level):
        self.level = level
        return

    def updateLabels(self, level):
        # Louvain algorithm labels community at different level (with dendrogram).
        # Here we want the community labels at a given level.
        level = int((len(self.dendrogram) - 1) * level)
        partition = community_louvain.partition_at_level(self.dendrogram, level)
        # Convert dictionary to numpy array
        self.labels = np.array(list(partition.values()))
        return

    def update(self, inputs, adj_mat=None):
        """Return the partition of the nodes at the given level.

        A dendrogram is a tree and each level is a partition of the graph nodes.
        Level 0 is the first partition, which contains the smallest communities,
        and the best is len(dendrogram) - 1.
        Higher the level is, bigger the communities are.
        """
        self.graph = nx.from_numpy_matrix(adj_mat)
        self.dendrogram = community_louvain.generate_dendrogram(self.graph)
        self.updateLabels(self.level)
        self.centroids = computeCentroids(inputs, self.labels)
        return


def computeCentroids(data, labels):
    n_clusters = len(np.unique(labels))
    return np.array([np.mean(data[labels == i], axis=0) for i in range(n_clusters)])


def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_sparse_matrix(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


class NB(object):
    def __init__(self, theta=None, scale_factor=1.0):
        super(NB, self).__init__()
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        y_pred = y_pred * self.scale_factor
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + self.eps)
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + self.eps))) + (
                y_true * (torch.log(theta + self.eps) - torch.log(y_pred + self.eps)))
        final = t1 + t2
        final = _nan2inf(final)
        if mean:
            final = torch.mean(final)
        return final


class ZINB(NB):
    def __init__(self, pi, ridge_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        nb_case = super().loss(y_true, y_pred, mean=False) - torch.log(1.0 - self.pi + eps)
        y_pred = y_pred * scale_factor
        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(self.pi + ((1.0 - self.pi) * zero_nb) + eps)
        result = torch.where(torch.lt(y_true, 1e-8), zero_case, nb_case)
        ridge = self.ridge_lambda * torch.square(self.pi)
        result += ridge
        if mean:
            result = torch.mean(result)
        result = _nan2inf(result)
        return result


class Colors():
    colors = [
        "#EEE8AA",
        "#ADD8E6",
        "#FFA500",
        "#FF8C00",
        "#808080",
        "#C0C0C0",
        "#008080",
        "#800080",
        "#808000",
        "#000080",
        "#008000",
        "#800000",
        "#00FFFF",
        "#FF00FF",
        "#FFFF00",
        "#0000FF",
        "#00FF00",
        "#FF0000",
        "01BBE7",
        "#FC4E27"]

def res_search_fixed_clus(cluster_type, adata, fixed_clus_count, increment=0.01):
    '''
                arg1(adata)[AnnData matrix]
                arg2(fixed_clus_count)[int]

                return:
                    resolution[int]
            '''
    if cluster_type == 'leiden':
        for res in sorted(list(np.arange(0.14, 2.5, increment))):#, reverse=True):
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            # print(res,' ' , count_unique_leiden)
            if count_unique_leiden == fixed_clus_count:
                cluster_labels = np.array(adata.obs['leiden'])
                flag=0
                break
            if count_unique_leiden > fixed_clus_count:
                cluster_labels = np.array(adata.obs['leiden'])
                flag=1
                break
    elif cluster_type == 'louvain':
        for res in sorted(list(np.arange(0.14, 2.5, increment))):#, reverse=True):
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique_louvain = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            # print(res,' ' , count_unique_louvain)
            if count_unique_louvain == fixed_clus_count:
                cluster_labels = np.array(adata.obs['louvain'])
                flag = 0
                break
            if count_unique_louvain > fixed_clus_count:
                cluster_labels = np.array(adata.obs['louvain'])
                flag = 1
                break
    return cluster_labels,flag


def PCA_process(X, nps):
    from sklearn.decomposition import PCA
    print('Shape of data to PCA:', X.shape)
    pca = PCA(n_components=nps)
    X_PC = pca.fit_transform(X)     #等价于pca.fit(X) pca.transform(X)
    print('Shape of data output by PCA:', X_PC.shape)
    print('PCA recover:', pca.explained_variance_ratio_.sum())
    return X_PC
def next_batch(X1, X2, batch_size):
    # generate next batch, just two views
    tot = X1.shape[0]
    total = math.ceil(tot / batch_size) - 1  # fix the last batch
    if tot % batch_size == 0:
        total += 1

    for i in range(int(total)):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        end_idx = min(tot, end_idx)
        batch_x1 = X1[start_idx: end_idx, ...]
        batch_x2 = X2[start_idx: end_idx, ...]
        yield batch_x1, batch_x2, (i + 1)
def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    # -*- coding : utf-8-*-
    # coding:unicode_escape

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata