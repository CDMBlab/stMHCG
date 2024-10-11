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
    path = "D:\Code\\first\stMHCG\data\DLPFC\\151510\\"
    labels_path = path + "metadata.tsv"

    labels = pd.read_table(labels_path, sep='\t')
    labels = labels["layer_guess_reordered"].copy()
    NA_labels = np.where(labels.isnull())
    labels = labels.drop(labels.index[NA_labels])
    ground = labels.copy()

    adata1 = sc.read_visium(path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata1.var_names_make_unique()

    obs_names = np.array(adata1.obs.index)
    positions = adata1.obsm['spatial']

    data = np.delete(adata1.X.toarray(), NA_labels, axis=0)
    obs_names = np.delete(obs_names, NA_labels, axis=0)
    positions = np.delete(positions, NA_labels, axis=0)

    adata = ad.AnnData(pd.DataFrame(data, index=obs_names, columns=np.array(adata1.var.index), dtype=np.float32))

    adata.var_names_make_unique()
    adata.obs['ground_truth'] = labels
    adata.obs['ground'] = ground
    adata.obsm['spatial'] = positions
    adata.obs['array_row'] = adata1.obs['array_row']
    adata.obs['array_col'] = adata1.obs['array_col']
    adata.uns['spatial'] = adata1.uns['spatial']
    adata.var['gene_ids'] = adata1.var['gene_ids']
    adata.var['feature_types'] = adata1.var['feature_types']
    adata.var['genome'] = adata1.var['genome']
    adata.var_names_make_unique()
    adata = normalize(adata, highly_genes=highly_genes)
    fadj = features_construct_graph(adata.X, k=k)
    sadj, graph_nei, graph_neg = spatial_construct_graph1(adata, radius=radius)
    cadata, oridata, augdata, stg = aug_construct_graph(adata,alpha=1)
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
    # = ['151508', '151509', '151510', '151669', '151670',
                 #'151671', '151672', '151673', '151676']
    datasets = ['151510']
    for i in range(len(datasets)):
        dataset = datasets[i]
        print(dataset)
        if not os.path.exists("../generate_data/DLPFC/"):
            os.mkdir("../generate_data/DLPFC/")
        savepath = "../generate_data/DLPFC/" + dataset + "/"
        config_file = './config/DLPFC.ini'
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        config = Config(config_file)
        adata = load_ST_file(dataset, config.fdim, config.k, config.radius)
        print("saving")
        adata.write(savepath + 'stMHCG.h5ad')
        print("done")
