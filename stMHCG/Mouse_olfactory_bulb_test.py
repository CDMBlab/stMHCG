from __future__ import division
from __future__ import print_function
from loss import *
from sklearn.metrics import silhouette_score, davies_bouldin_score
import torch.optim as optim
from utils import *
from models import stMHCG
import os
import argparse
from config import Config
import pandas as pd


def load_data(dataset):
    print("load data")
    path = "../generate_data/" + dataset + "/stMHCG_MOB.h5ad"
    adata = sc.read_h5ad(path)
    dense_array = adata.X.toarray()
    print(adata)
    features = torch.FloatTensor(dense_array)
    fadj = adata.obsm['fadj']
    sadj = adata.obsm['sadj']
    stg = adata.obsm['stg']
    stg = torch.Tensor(stg)
    features_aug = adata.obsm['augdata']
    features_aug = torch.Tensor(features_aug)
    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    graph_nei = torch.LongTensor(adata.obsm['graph_nei'])
    graph_neg = torch.LongTensor(adata.obsm['graph_neg'])
    print("done")
    return features, nsadj, nfadj, graph_nei, graph_neg,stg,features_aug,adata


def train():
    model.train()
    optimizer.zero_grad()
    print(features.shape)
    print(features_aug.shape)
    com1, com2, com3, emb, pi, disp, mean, y1, p1, y2, p2, y3, p3, y4, p4, y, p = model(features, features_aug, sadj,
                                                                                        fadj, stg)
    zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features, mean, mean=True)
    reg_loss = regularization_loss(emb, graph_nei, graph_neg)
    ##########add loss###################
    # cluster contrastive loss
    cluster_loss = criterion_cluster(y3, y1) + criterion_cluster(y2, y1) + criterion_cluster(y2, y3)
    cluster_loss = cluster_loss * 0.01
    print("cluster_loss", cluster_loss)
    # high confidence loss
    y_max = torch.maximum(y1, y2)
    y_max = torch.maximum(y_max, y)
    y_max = target_l2(y_max)
    y = torch.where(y < EPS, torch.tensor([EPS], device=y.device), y)
    hc_loss1 = F.kl_div(y.log(), y_max.detach(), reduction='batchmean')
    y_max = torch.maximum(y2, y3)
    y_max = torch.maximum(y_max, y)
    y_max = target_l2(y_max)
    y = torch.where(y < EPS, torch.tensor([EPS], device=y.device), y)
    hc_loss2 = F.kl_div(y.log(), y_max.detach(), reduction='batchmean')
    y_max = torch.maximum(y3, y1)
    y_max = torch.maximum(y_max, y)
    y_max = target_l2(y_max)
    y = torch.where(y < EPS, torch.tensor([EPS], device=y.device), y)
    hc_loss3 = F.kl_div(y.log(), y_max.detach(), reduction='batchmean')
    hc_loss = hc_loss1 + hc_loss2 + hc_loss3
    hc_loss = hc_loss * 0.0001
    print("hc_loss", hc_loss)
    total_loss = config.alpha * zinb_loss + config.gamma * reg_loss + cluster_loss + hc_loss
    emb = pd.DataFrame(emb.cpu().detach().numpy()).fillna(0).values
    mean = pd.DataFrame(mean.cpu().detach().numpy()).fillna(0).values
    total_loss.backward()
    optimizer.step()
    y = y.data.cpu().numpy().argmax(1)
    return emb, mean, zinb_loss, reg_loss, cluster_loss, total_loss, y

if __name__ == "__main__":
    # parse = argparse.ArgumentParser()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']  # 列出你希望使用的serif字体
    # 设置字体大小
    plt.rcParams['font.size'] = 10
    datasets = ['MOB']
    for i in range(len(datasets)):
        dataset = datasets[i]
        savepath = './results/' + dataset + '/'
        config_file = './config/' + 'Mouse_Olfactory.ini'
        # hire_path = '../data/' + dataset + '/spatial/crop1.png'
        # img = plt.imread(hire_path)

        if not os.path.exists(savepath):
            os.mkdir(savepath)
        print(dataset)

        features, sadj, fadj, graph_nei, graph_neg,stg,features_aug,adata= load_data(dataset)
        print(features.shape)
        print(features_aug.shape)
        print(sadj.shape)
        print(fadj.shape)
        print(stg.shape)
        config = Config(config_file)
        cuda = False
        use_seed = not config.no_seed
        config.n = features.shape[0]
        config.class_num = 7

        # if cuda:
        #     features = features.cuda()
        #     features_aug=features_aug.cuda()
        #     sadj = sadj.cuda()
        #     fadj = fadj.cuda()
        #     stg = stg.cuda()
        #     graph_nei = graph_nei.cuda()
        #     graph_neg = graph_neg.cuda()

        config.epochs = config.epochs + 1
        import random

        np.random.seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        os.environ['PYTHONHASHSEED'] = str(config.seed)
        if not config.no_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

        print(dataset, ' ', config.lr, ' ', config.alpha, ' ', config.beta, ' ', config.gamma)
        model = Spatial_MGCN(nfeat=config.fdim,
                            nhid1=config.nhid1,
                            nhid2=config.nhid2,
                            dropout=config.dropout,
                            class_num= config.class_num)
        criterion_instance = InstanceLoss(config.epochs, 1.0, cuda)
        criterion_cluster = ClusterLoss(config.class_num, 0.5, cuda)
        if cuda:
            model.cuda()
            criterion_instance.cuda()
            criterion_cluster.cuda()
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        epoch=1
        for epoch in range(config.epochs):
            print(epoch)
            epoch=epoch+1
            emb, mean, zinb_loss, reg_loss, con_loss, total_loss, y = train()

            # print(dataset, ' epoch: ', epoch, ' zinb_loss = {:.2f}'.format(zinb_loss),
            #       ' reg_loss = {:.2f}'.format(reg_loss), ' con_loss = {:.2f}'.format(con_loss),
            #       ' total_loss = {:.2f}'.format(total_loss))
        adata.obsm['emb'] = emb
        emb_array = adata.obsm['emb']
        emb_df = pd.DataFrame(emb_array)
        emb_df.to_csv(savepath + 'stMHCG_emb.csv', header=False, index=False)
        sc.pp.neighbors(adata, use_rep='emb')
        sc.tl.umap(adata)
        sc.tl.louvain(adata, resolution=0.8)
        print(adata)
        #sc.tl.leiden(adata, resolution=0.8)
        adata.obs['louvain_clusters'] = adata.obs['louvain']
        #adata = mclust_R(adata, num_cluster=config.class_num)
        #mclust_array = adata.obs['mclust']
        #mclust_df = pd.DataFrame(mclust_array)
        #mclust_df.to_csv(savepath + 'stMHCG_idx.csv', header=None, index=None)
        louvain_clusters = adata.obs['louvain_clusters']
        louvain_df = pd.DataFrame(louvain_clusters)
        louvain_df.to_csv(savepath + 'stMHCG_louvain_idx.csv', header=None, index=None)
        silhouette_avg = silhouette_score(adata.obsm['emb'], adata.obs['louvain'])
        db_index = davies_bouldin_score(adata.obsm['emb'], adata.obs['louvain'])
        print('silhouette_avg', silhouette_avg)
        print('db_index', db_index)
        hire_path = '../data/MOB/spatial/crop1.png'
        img = plt.imread(hire_path)
        pl = ['#1f77b4ff', '#ff7f0eff', '#2ca02cff', '#d62728ff', '#9467bdff', '#8c564bff',
              '#e377c2ff']

        plt.axis('off')

        ax = plt.gca()
        ax.imshow(img, extent=[5740, 12410, 9750, 15420])
        sc.pl.embedding(adata, basis="spatial", color="louvain", s=30, show=False,
                        title='stMHCG', palette=pl, ncols=2, vmin=0, vmax='p99.2')
        plt.savefig(savepath + 'stMHCG.jpg', bbox_inches='tight', dpi=600)
        plt.show()
        plt.close()
        adata.layers['X'] = adata.X
        adata.layers['mean'] = mean
        adata.write(savepath + 'stMHCG_clustering.h5ad')

        adata.obs['louvain'].replace(1, 'GL', inplace=True)
        adata.obs['louvain'].replace(2, 'GCL', inplace=True)
        adata.obs['louvain'].replace(3, 'IPL', inplace=True)
        adata.obs['louvain'].replace(4, 'ONL', inplace=True)
        adata.obs['louvain'].replace(5, 'EPL', inplace=True)
        adata.obs['louvain'].replace(6, 'RMS', inplace=True)
        adata.obs['louvain'].replace(7, 'MCL', inplace=True)

        n_type = config.class_num
        zeros = np.zeros([adata.n_obs, n_type])
        cell_type = list(adata.obs['louvain'].unique())
        cell_type = [str(s) for s in cell_type]
        cell_type.sort()
        print(n_type)  # 查看 n_type 的值
        print(cell_type)  # 查看 cell_type 列表的长度

        matrix_cell_type = pd.DataFrame(zeros, index=adata.obs_names, columns=cell_type)
        for cell in list(adata.obs_names):
            ctype = adata.obs.loc[cell, 'louvain']
            matrix_cell_type.loc[cell, str(ctype)] = 1

        adata.obs[matrix_cell_type.columns] = matrix_cell_type.astype(str)
        savepath = savepath + 'align/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        cell_types = ['EPL', 'GCL', 'GL', 'IPL', 'MCL', 'ONL', 'RMS', ]
        for j in range(len(cell_types)):
            sc.pl.embedding(adata, basis="spatial", color=cell_types[j], s=10, palette=['gray', pl[j]],
                            show=False, vmin=0, vmax='p99.2')
            plt.savefig(savepath + cell_types[j] + '.jpg',
                        bbox_inches='tight', dpi=600)
            plt.show()