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
#小鼠大脑前部区域数据集53个区域，生成的数据不包含标签

def normalize(adata, highly_genes=3000):
    print("start select HVGs")
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=highly_genes)
    adata = adata[:, adata.var['highly_variable']].copy()
    adata.X = adata.X / np.sum(adata.X, axis=1).reshape(-1, 1) * 10000
    sc.pp.scale(adata, zero_center=False, max_value=10)
    return adata


def load_ST_file(dataset, highly_genes, k, radius):
    path = "../data/Mouse_Brain_Anterior/" +  "/"
    labels_path = path + "metadata.tsv"

    labels = pd.read_table(labels_path, sep='\t')
    labels = labels["ground_truth"].copy()
    NA_labels = np.where(labels.isnull())
    labels = labels.drop(labels.index[NA_labels])
    labels = labels.astype(str)
    labels = [str(label) for label in labels]
    ground = labels.copy()
    print(type(ground))
    for label in labels:
        print(type(label))

    # ground.replace('AOB::GI', '1', inplace=True)
    # ground.replace('AOB::Gr', '2', inplace=True)
    # ground.replace('AOB::MI', '3', inplace=True)
    # ground.replace('AOE', '4', inplace=True)
    # ground.replace('AON::L1_1', '5', inplace=True)
    # ground.replace('AON::L1_2', '6', inplace=True)
    # ground.replace('AON::L2', '7', inplace=True)
    # ground.replace('AcbC', '8', inplace=True)
    # ground.replace('AcbSh', '9', inplace=True)
    # ground.replace('CC', '10', inplace=True)
    # ground.replace('CPu', '11', inplace=True)
    # ground.replace('CI', '12', inplace=True)
    # ground.replace('En', '13', inplace=True)
    # ground.replace('FRP::L1', '14', inplace=True)
    # ground.replace('FRP::L2/3', '15', inplace=True)
    # ground.replace('Fim', '16', inplace=True)
    # ground.replace('Ft', '17', inplace=True)
    # ground.replace('HY::LPO', '18', inplace=True)
    # ground.replace('Io', '19', inplace=True)
    # ground.replace('LV', '20', inplace=True)
    # ground.replace('MO::L1', '21', inplace=True)
    # ground.replace('MO::L2/3', '22', inplace=True)
    # ground.replace('MO::L5', '23', inplace=True)
    # ground.replace('MO::L6', '24', inplace=True)
    # ground.replace('MOB::GI_1', '25', inplace=True)
    # ground.replace('MOB::GI_2', '26', inplace=True)
    # ground.replace('MOB::Gr', '27', inplace=True)
    # ground.replace('MOB::MI', '28', inplace=True)
    # ground.replace('MOB::OpI', '29', inplace=True)
    # ground.replace('MOB::IpI', '30', inplace=True)
    # ground.replace('Not_annotated', '31', inplace=True)
    # ground.replace('ORB::L1', '32', inplace=True)
    # ground.replace('ORB::L2/3', '33', inplace=True)
    # ground.replace('ORB::L5', '34', inplace=True)
    # ground.replace('ORB::L6', '35', inplace=True)
    # ground.replace('OT::MI', '36', inplace=True)
    # ground.replace('OT::PI', '37', inplace=True)
    # ground.replace('OT::PoL', '38', inplace=True)
    # ground.replace('Or', '39', inplace=True)
    # ground.replace('PIR', '40', inplace=True)
    # ground.replace('Pal::GPi', '41', inplace=True)
    # ground.replace('Pal::MA', '42', inplace=True)
    # ground.replace('Pal::NDB', '43', inplace=True)
    # ground.replace('Pal::SI', '44', inplace=True)
    # ground.replace('Py', '45', inplace=True)
    # ground.replace('SLu', '46', inplace=True)
    # ground.replace('SS::L1', '47', inplace=True)
    # ground.replace('SS::L2/3', '48', inplace=True)
    # ground.replace('SS::L5', '49', inplace=True)
    # ground.replace('SS::L6', '50', inplace=True)
    # ground.replace('St', '51', inplace=True)
    # ground.replace('TH::RT', '52', inplace=True)


    adata1 = sc.read_visium(path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata1.var_names_make_unique()

    obs_names = np.array(adata1.obs.index)
    positions = adata1.obsm['spatial']

    data = np.delete(adata1.X.toarray(), NA_labels, axis=0)
    obs_names = np.delete(obs_names, NA_labels, axis=0)
    positions = np.delete(positions, NA_labels, axis=0)

    adata = ad.AnnData(pd.DataFrame(data, index=obs_names, columns=np.array(adata1.var.index), dtype=np.float32))
    #print(labels.dtype)
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
    #datasets = ['151507']
    datasets = ['Mouse_Brain_Anterior']
    for i in range(len(datasets)):
        dataset = datasets[i]
        print(dataset)
        if not os.path.exists("../generate_data/Mouse_Brain_Anterior/"):
            os.mkdir("../generate_data/Mouse_Brain_Anterior/")
        savepath = "../generate_data/Mouse_Brain_Anterior/" + dataset + "/"
        config_file = './config/DLPFC.ini'
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        config = Config(config_file)
        adata = load_ST_file(dataset, config.fdim, config.k, config.radius)
        print("saving")
        adata.write(savepath + 'stMHCG_1000.h5ad')
        print("done")
