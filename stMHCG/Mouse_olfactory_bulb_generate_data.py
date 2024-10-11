from __future__ import division
from __future__ import print_function

import json


import torch
import os
import argparse
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import Config
from utils import features_construct_graph, spatial_construct_graph,aug_construct_graph
import anndata as ad
from matplotlib.image import imread
from pathlib import Path


def normalize(adata, highly_genes=3000):
    print("start select HVGs")
    # sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=highly_genes)
    adata = adata[:, adata.var['highly_variable']].copy()
    sums = np.sum(adata.X, axis=1)
    eps = 1e-10  # 一个小的数值，用于避免除以0
    adata.X[sums == 0] = eps  # 将和为0的行替换为一个小的数值
    adata.X = adata.X / np.sum(adata.X, axis=1).reshape(-1, 1) * 10000
    sc.pp.scale(adata, zero_center=False, max_value=10)
    return adata


def load_ST_file(dataset, highly_genes, k, k1):

    counts_file = os.path.join('../data/MOB/spatial/RNA_counts.tsv')
    coor_file = os.path.join('../data/MOB/spatial/position.tsv')
    #counts_file = os.path.join('D:\Code\\first\A_baseline\data\MOB\RNA_counts.tsv')
    #coor_file = os.path.join('D:\Code\\first\A_baseline\data\MOB\position.tsv')
    counts = pd.read_csv(counts_file, sep='\t', index_col=0)
    coor_df = pd.read_csv(coor_file, sep='\t')
    print(counts.shape, coor_df.shape)
    counts.columns = ['Spot_' + str(x) for x in counts.columns]
    coor_df.index = coor_df['label'].map(lambda x: 'Spot_' + str(x))
    coor_df = coor_df.loc[:, ['x', 'y']]
    print(coor_df.head())
    adata = sc.AnnData(counts.T)
    adata = normalize(adata, highly_genes=highly_genes)
    coor_df = coor_df.loc[adata.obs_names, ['y', 'x']]
    adata.obsm["spatial"] = coor_df.to_numpy()
    print(adata)
    # sc.pp.calculate_qc_metrics(adata, inplace=True)
    # plt.rcParams["figure.figsize"] = (5, 4)
    # sc.pl.embedding(adata, basis="spatial", color="n_genes_by_counts")
    # plt.title("")
    # plt.axis('off')
    used_barcode = pd.read_csv('../data/MOB/used_barcodes.txt', sep='\t',
                               header=None)
    used_barcode = used_barcode[0]
    adata = adata[used_barcode,]
    print(adata)
    # plt.rcParams["figure.figsize"] = (5, 4)
    # sc.pl.embedding(adata, basis="spatial", color="n_genes_by_counts", show=False)
    # plt.title("")
    # plt.axis('off')
    adata.var_names_make_unique()
    print('After flitering: ', adata.shape)
    X = adata.X
    fadj = features_construct_graph(X, k=k)
    sadj, graph_nei, graph_neg = spatial_construct_graph(adata.obsm["spatial"], k=k1)
    cadata, oridata, augdata, stg = aug_construct_graph(adata, alpha=1)
    print(augdata.shape)
    print(oridata.shape)
    print(adata.shape)
    print(stg.shape)
    print(fadj.shape)
    print(sadj.shape)
    adata.obsm["fadj"] = fadj
    adata.obsm["sadj"] = sadj
    adata.obsm["stg"] = stg
    adata.obsm["oridata"] = oridata
    adata.obsm["augdata"] = augdata
    adata.obsm["graph_nei"] = graph_nei.numpy()
    adata.obsm["graph_neg"] = graph_neg.numpy()
    adata.var_names_make_unique()
    # path = "../data/" + dataset + "/"
    # data_path = path + "filtered_feature_bc_matrix.h5ad"
    # positions_path = path + "spatial/tissue_positions_list.csv"
    # savepath = './result/Mouse_Olfactory/Raw/'
    # hires_image = path + 'spatial/crop1.png'
    # #scalefactors_json_file = path + 'spatial/scalefactors_json.json'
    # if not os.path.exists(savepath):
    #     os.mkdir(savepath)
    #
    # annData = sc.read_h5ad(data_path)
    # adata = normalize(annData, highly_genes=highly_genes)
    #
    # positions = pd.read_csv(positions_path, sep=',')
    # index_labels = np.array(annData.obs.index)
    # for i in range(len(index_labels)):
    #     index_labels[i] = index_labels[i][5:]
    # index_labels = index_labels.astype(int)
    # position = pd.DataFrame(columns=['y', 'x'])
    # for i in range(len(index_labels)):
    #     position.loc[i] = positions[positions['barcode'] == index_labels[i]].values[0][5:7]
    # positions = np.array(position, dtype=float)
    # positions[:, [0, 1]] = positions[:, [1, 0]]
    #
    # barcodes = np.array(adata.obs.index)
    # names = np.array(adata.var.index)
    # # X = adata.X
    # X = np.nan_to_num(adata.X, nan=0)
    #
    # n = len(positions)
    # index = range(0, n, 15)
    # X = np.delete(X, index, axis=0)
    # positions = np.delete(positions, index, axis=0)
    # barcodes = np.delete(barcodes, index, axis=0)
    #
    # fadj = features_construct_graph(X, k=k)
    # sadj, graph_nei, graph_neg = spatial_construct_graph(positions, k=k1)
    #
    # adata = ad.AnnData(pd.DataFrame(X, index=barcodes, columns=names))  # , dtype=adata.dtype)
    # adata.var_names_make_unique()
    # adata.obs['barcodes'] = barcodes
    # adata.var['names'] = names
    #
    # adata.obsm["spatial"] = positions
    #
    # adata.obsm["fadj"] = fadj
    # adata.obsm["sadj"] = sadj
    # adata.obsm["graph_nei"] = graph_nei.numpy()
    # adata.obsm["graph_neg"] = graph_neg.numpy()
    #
    # adata.uns["spatial"] = {}
    # adata.uns["spatial"][dataset] = {}
    # adata.uns["spatial"][dataset]['images'] = {}
    # adata.uns["spatial"][dataset]['images']['hires'] = imread(hires_image)
    # # adata.uns["spatial"][dataset]['scalefactors'] = json.loads(Path(scalefactors_json_file).read_bytes())
    # adata.var_names_make_unique()
    return adata


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    datasets = ['MOB']
    for i in range(len(datasets)):
        dataset = datasets[i]
        print(dataset)
        if not os.path.exists("../generate_data/"):
            os.mkdir("../generate_data/")
        savepath = "../generate_data/" + dataset + "/"
        config_file = './config/' +  'Mouse_Olfactory.ini'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        config = Config(config_file)
        adata = load_ST_file(dataset, config.fdim, config.k, config.radius)
        print("saving")
        adata.write(savepath + 'stMHCG_MOB.h5ad')
        print("done")

