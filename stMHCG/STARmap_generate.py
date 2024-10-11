from __future__ import division
from __future__ import print_function

from utils import features_construct_graph, spatial_construct_graph1,aug_construct_graph
import os
import argparse
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from config import Config


def normalize(adata, highly_genes=3000):
    print("start select HVGs")
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=highly_genes)
    adata = adata[:, adata.var['highly_variable']].copy()
    adata.X = adata.X / np.sum(adata.X, axis=1).reshape(-1, 1) * 10000
    sc.pp.scale(adata, zero_center=False, max_value=10)
    return adata


def load_ST_file(dataset, highly_genes, k, radius):
    adata = sc.read('D:\Code\Spatial-MGCN-master\data\STARmap\STARmap_20180505_BY3_1k.h5ad')
    adata.var_names_make_unique()
    obs_names = np.array(adata.obs.index)
    positions = adata.obsm['spatial']
    adata.var_names_make_unique()
    adata = normalize(adata, highly_genes=highly_genes)
    fadj = features_construct_graph(adata.X, k=k)
    sadj, graph_nei, graph_neg = spatial_construct_graph1(adata, radius=radius)
    cadata, oridata, augdata, stg = aug_construct_graph(adata, alpha=1)
    print(adata.shape)
    print(cadata.shape)
    print(oridata.shape)
    print(augdata.shape)
    adata.obsm["fadj"] = fadj
    adata.obsm["sadj"] = sadj
    adata.obsm["stg"] = stg
    adata.obsm["oridata"] = oridata
    adata.obsm["augdata"] = augdata
    adata.obsm["graph_nei"] = graph_nei.numpy()
    adata.obsm["graph_neg"] = graph_neg.numpy()
    adata.var_names_make_unique()
    return adata


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # datasets = ['151507', '151508', '151509', '151510', '151669', '151670',
    #             '151671', '151672', '151673', '151674', '151675', '151676']
    datasets = ['STARmap']
    for i in range(len(datasets)):
        dataset = datasets[i]
        print(dataset)
        if not os.path.exists("../generate_data/STARmap/"):
            os.mkdir("../generate_data/STARmap/")
        savepath = "../generate_data/STARmap/" + dataset + "/"
        config_file = './config/DLPFC.ini'
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        config = Config(config_file)
        adata = load_ST_file(dataset, config.fdim, config.k, config.radius)
        print("saving")
        adata.write(savepath + 'stMHCG.h5ad')
        print("done")
