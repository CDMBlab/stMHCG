from __future__ import division
from __future__ import print_function
from sklearn.utils import shuffle
import torch.optim as optim
from loss import *
from utils import *
from models import stMHCG
import os
import argparse
from config import Config
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt


def load_data(dataset):
    print("load data:")
    path = "../generate_data/" + dataset + "/stMHCG.h5ad"
    adata = sc.read_h5ad(path)
    sparse_matrix = adata.X
    if not isinstance(sparse_matrix, csr_matrix):
        sparse_matrix = sparse_matrix.tocsr()
    dense_matrix = sparse_matrix.toarray()
    features = torch.FloatTensor(dense_matrix)
    features_aug = adata.obsm['augdata']
    features_aug = torch.Tensor(features_aug)
    features_ori =adata.obsm['oridata']
    features_ori = torch.Tensor(features_ori)
    labels = adata.obs['ground']
    fadj = adata.obsm['fadj']
    sadj = adata.obsm['sadj']
    stg = adata.obsm['stg']
    stg = torch.Tensor(stg)
    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    graph_nei = torch.LongTensor(adata.obsm['graph_nei'])
    graph_neg = torch.LongTensor(adata.obsm['graph_neg'])
    print("done")
    return adata, features, features_aug, labels, nfadj, nsadj, stg, graph_nei, graph_neg,features_ori

def train():
    model.train()
    optimizer.zero_grad()
    # com1, com2, com3,emb, pi, disp, mean,z1,z2,z3,z4,y1,p1,y2,p2,y3,p3,y4,p4
    com1, com2, com3, emb, pi, disp, mean, y1, p1, y2, p2, y3, p3, y4, p4, y, p = model(features, features_aug, sadj,
                                                                                        fadj, stg)
    zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features, mean, mean=True)
    reg_loss = regularization_loss(emb, graph_nei, graph_neg)
    # con_loss = consistency_loss(com1, com2,com3)
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
    #####################################
    total_loss = config.alpha * zinb_loss + config.gamma * reg_loss + cluster_loss + hc_loss
    # total_loss = config.alpha * zinb_loss + config.beta * con_loss + config.gamma * reg_loss+cluster_loss+hc_loss
    # total_loss = config.alpha * zinb_loss + config.beta * con_loss + cluster_loss + hc_loss
    emb = pd.DataFrame(emb.cpu().detach().numpy()).fillna(0).values
    mean = pd.DataFrame(mean.cpu().detach().numpy()).fillna(0).values
    total_loss.backward()
    optimizer.step()
    y = y.data.cpu().numpy().argmax(1)
    return emb, mean, zinb_loss, reg_loss, cluster_loss, total_loss, y


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']  # 列出你希望使用的serif字体
    # 设置字体大小
    plt.rcParams['font.size'] = 10
    datasets = ['Human_Breast_Cancer']

    for i in range(len(datasets)):
        dataset = datasets[i]
        path = './results/' + dataset + '/'
        config_file = './config/' + dataset + '.ini'
        if not os.path.exists(path):
            os.mkdir(path)
        print(dataset)
        adata, features, features_aug, labels, fadj, sadj, stg, graph_nei, graph_neg ,features_ori= load_data(dataset)

        config = Config(config_file)
        cuda = not config.no_cuda and torch.cuda.is_available()
        use_seed = not config.no_seed

        _, ground = np.unique(np.array(labels, dtype=str), return_inverse=True)
        ground = torch.LongTensor(ground)
        config.n = len(ground)
        config.class_num = len(ground.unique())

        savepath = './results/Human_Breast_Cancer/'
        plt.rcParams["figure.figsize"] = (4, 3)

        print(adata)
        title = "Manual annotation"
        sc.pl.spatial(adata, img_key="hires", color=['ground_truth'],title=title, show=False)
        plt.savefig(savepath + dataset + '.jpg', bbox_inches='tight', dpi=600)
        plt.show()

        if cuda:
            features = features.cuda()
            features_ori=features_ori.cuda()
            features_aug = features_aug.cuda()
            sadj = sadj.cuda()
            fadj = fadj.cuda()
            stg = stg.cuda()
            graph_nei = graph_nei.cuda()
            graph_neg = graph_neg.cuda()

        config.epochs = config.epochs + 1

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
                             class_num=config.class_num
                             )
        criterion_instance = InstanceLoss(config.epochs, 1.0, cuda)
        criterion_cluster = ClusterLoss(config.class_num, 0.5, cuda)
        if cuda:
            model.cuda()
            criterion_instance.cuda()
            criterion_cluster.cuda()
        optimizer = optim.Adam(model.parameters(), lr=config.lr,
                               weight_decay=config.weight_decay)
        epoch_max = 0
        ari_max = 0
        idx_max = []
        mean_max = []
        emb_max = []
        nmi_max=0
        for epoch in range(config.epochs):
            emb, mean, zinb_loss, reg_loss, con_loss, total_loss, y = train()
            print(dataset, ' epoch: ', epoch, ' zinb_loss = {:.2f}'.format(zinb_loss),
                  ' reg_loss = {:.2f}'.format(reg_loss), ' con_loss = {:.2f}'.format(con_loss),
                  ' total_loss = {:.2f}'.format(total_loss))
            kmeans = KMeans(n_clusters=config.class_num).fit(emb)
            idx = kmeans.labels_
            ari_res = metrics.adjusted_rand_score(labels, idx)
            nmi_res = metrics.normalized_mutual_info_score(labels, idx)
            if ari_res > ari_max:
                ari_max = ari_res
                epoch_max = epoch
                idx_max = idx
                mean_max = mean
                emb_max = emb
            if nmi_res > nmi_max:
                nmi_max = nmi_res
                epoch_max_nmi = epoch
                idx_max_nmi = idx
        print(dataset, 'ARI', ari_max, 'NMI', nmi_max)

        adata.obs['idx'] = idx_max.astype(str)
        adata.obsm['emb'] = emb_max
        adata.obsm['mean'] = mean_max
        title = 'stMHCG: ARI={:.3f} NMI={:.3f}'.format(ari_max,nmi_max)
        if config.gamma == 0:
            pd.DataFrame(emb_max).to_csv(savepath + 'stMHCG_no_emb.csv', header=None, index=None)
            pd.DataFrame(idx_max).to_csv(savepath + 'stMHCG_no_idx.csv', header=None, index=None)
            sc.pl.spatial(adata, img_key="hires", color=['idx'], title=title,show=False)
            plt.savefig(savepath + 'stMHCG_no.jpg', bbox_inches='tight', dpi=600)
            plt.show()
            sc.pp.neighbors(adata, use_rep='mean')
            sc.tl.umap(adata)
            sc.pl.umap(adata, color=['idx'], legend_loc='on data', s=10,show=False,title='stMHCG')
            plt.savefig(savepath + 'stMHCG_2.jpg', bbox_inches='tight', dpi=600)
        else:
            pd.DataFrame(emb_max).to_csv(savepath + 'stMHCG_emb.csv', header=None, index=None)
            pd.DataFrame(idx_max).to_csv(savepath + 'stMHCG_idx.csv', header=None, index=None)
            sc.pl.spatial(adata, img_key="hires", color=['idx'], title=title,show=False)
            adata.layers['X'] = adata.X
            adata.layers['mean'] = mean_max
            plt.savefig(savepath + 'stMHCG.jpg', bbox_inches='tight', dpi=600)
            plt.show()
            adata.write(savepath + 'stMHCG.h5ad')
            sc.pp.neighbors(adata, use_rep='mean')
            sc.tl.umap(adata)
            sc.pl.umap(adata, color=['idx'], legend_loc='on data', s=10,show=False,title='stMHCG')
            plt.savefig(savepath + 'stMHCG_2.jpg', bbox_inches='tight', dpi=600)


