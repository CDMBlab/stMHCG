import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch
from torch.nn.functional import normalize

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class decoder(torch.nn.Module):
    def __init__(self, nfeat, nhid1, nhid2):
        super(decoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1),
            torch.nn.BatchNorm1d(nhid1),
            torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1, nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, disp, mean]


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class InstanceProject(nn.Module):
    def __init__(self, latent_dim):
        super(InstanceProject, self).__init__()
        self._latent_dim = latent_dim
        self.instance_projector = nn.Sequential(
            nn.Linear(self._latent_dim, self._latent_dim),
            nn.BatchNorm1d(self._latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.instance_projector(x)
class ClusterProject(nn.Module):
    def __init__(self, latent_dim, n_clusters):
        super(ClusterProject, self).__init__()
        self._latent_dim = latent_dim
        self._n_clusters = n_clusters
        self.cluster_projector = nn.Sequential(
            nn.Linear(self._latent_dim, self._latent_dim),
            nn.BatchNorm1d(self._latent_dim),
            nn.ReLU(),
            # nn.Linear(self._latent_dim, self._latent_dim),
            # nn.BatchNorm1d(self._latent_dim),
            # nn.ReLU(),
        )
        self.cluster = nn.Sequential(
            nn.Linear(self._latent_dim, self._n_clusters),
            # nn.BatchNorm1d(self._n_clusters),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        z = self.cluster_projector(x)
        y = self.cluster(z)
        return y, z
class stMHCG(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout,class_num):
        super(stMHCG, self).__init__()
        self.SGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.FGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.AGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.ZINB = decoder(nfeat, nhid1, nhid2)
        self.dropout = dropout
        self.att = Attention(nhid2)
        self.MLP = nn.Sequential(
            nn.Linear(nhid2, nhid2)
        )
        self.instance_projector1 = InstanceProject(nhid2)
        self.instance_projector2 = InstanceProject(nhid2)
        self.instance_projector3 = InstanceProject(nhid2)
        self.instance_projector4 = InstanceProject(nhid2)
        self.cluster = ClusterProject(nhid2, class_num)

    def forward(self, x, x_aug, sadj, fadj,stg):
        #print(x.shape)
        #print(x_aug.shape)
        #print(sadj.shape)
        #print(fadj.shape)
        #print(stg.shape)
        # com1, com2, com3,emb, pi, disp, mean = model(features, features_aug,sadj, fadj,stg)
        emb1 = self.SGCN(x, sadj)  # Spatial_GCN
        com1 = self.CGCN(x, sadj)  # Co_GCN
        com2 = self.CGCN(x, fadj)  # Co_GCN
        emb2 = self.FGCN(x, fadj)  # Feature_GCN
        emb3 = self.AGCN(x_aug, stg)  # 增强_GCN
        com3 = self.CGCN(x_aug, stg)  # Co_GCN
        com = emb1+emb2+emb3
        emb = torch.stack([emb1, com / 3, emb2, emb3], dim=1)
        emb, att = self.att(emb)
        emb = self.MLP(emb)
        [pi, disp, mean] = self.ZINB(emb)
        # z1 = normalize(self.instance_projector1(emb1), dim=1)
        # z2 = normalize(self.instance_projector2(emb2), dim=1)
        # z3 = normalize(self.instance_projector3(emb3), dim=1)
        # z4 = normalize(self.instance_projector4(com), dim=1)
        y1, p1 = self.cluster(com1)
        y2, p2 = self.cluster(com2)
        y3, p3 = self.cluster(com3)
        y4, p4 = self.cluster(com)
        y, p = self.cluster(emb)
        return com1, com2, com3,emb, pi, disp, mean,y1,p1,y2,p2,y3,p3,y4,p4,y,p
