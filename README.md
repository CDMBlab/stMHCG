# stMHCG: stMHCG: high-confidence multi-view clustering leverages identification of spatial domains form spatially resolved transcriptomics
![model](https://github.com/CDMBlab/stMHCG/blob/main/49262913f09e3f54a6170ce955d2380.png)
## Requirements 
Python==3.8.19
numpy==1.24.4
pandas==2.0.3
scipy==1.10.1
scikit-learn==1.3.2
pytorch==2.0.0+cu118
torch-cluster==1.6.1+pt20cu118
torch_geometric==2.5.2
torch-scatter==2.1.1+pt20cu118
torch-sparse==0.6.17+pt20cu118
torch-spline-conv==1.2.2+pt20cu118
torchvision==0.15.1+cu118
matplotlib==3.7.5
louvain==0.8.2
rpy2==3.4.5

## Example

### 1 Raw data 

Raw data should be placed in the folder ***data***.

we put the DLPFC dataset in ***data/DLPFC***. Need to be extracted firstly.

For 10x Visium datasets, files should be put in the same structure with that provided by 10x website. Taking DLPFC for instance:

> data/DLPFC/151510/ 
  >> spatial/  # The folder where files for spatial information can be found 
  
  >> metadata.tsv # mainly for annotation
  
  >> filtered_feature_bc_matrix.h5 # gene expression data


### 2 Configuration

The meaning of each argument in ***config.py*** is listed below.

**--epochs**: the number of training epochs.

**--lr**: the learning rate.

**--weight_decay**: the weight decay.

**--k**: the k-value of the k-nearest neighbor graph, which should be within {8...15}.

**--radius**：the spatial location radius.

**--nhid1**: the dimension of the first hidden layer. 

**--nhid2**: the dimension of the second hidden layer. 

**--dropout**: the dropout rate.

**--no_cuda**: whether to use GPU.

**--no_seed**: whether to take the random parameter.

**--seed**: the random parameter.

**--fdim**: the number of highly variable genes selected.


### 3 Data Preprocessing and Graph Construction

Run ***stMHCG/DLPFC_generate_data.py*** to preprocess the raw DLPFC data:

`python DLPFC_generate_data.py`

Augments:

**--savepath**: the path to save the generated file.

For dealing other ST datasets, please modify the data name. 


### 4 Usage

For training stMHCG model, run

'python DLPFC_test.py'

All results are saved in the result folder. We provide our results in the folder ***result*** for taking further analysis. 

(1) The cell clustering labels are saved in ***stMHCG_idx.csv***, where the first column refers to cell index, and the last column refers to cell cluster label. 

(2) The trained embedding data are saved in ***stMHCG_emb.csv***.

For Human_Breast_Cancer,Mouse_brain_anterior,STARmap and mouse_olfactory_bulb datasets, the running process is the same as above. You just need to run the command:

'python Human_breast_cancer_test.py'

'python Mouse_olfactory_bulb_test.py'

'python Mouse_brain_anterior_test.py'

'python STARmap_test.py'
## Download all datasets used in stMHCG:

The datasets used in this paper can be downloaded from the following websites. Specifically,
(1) The LIBD human dorsolateral prefrontal cortex (DLPFC) dataset http://spatial.libd.org/spatialLIBD
(2) the processed Stereo-seq dataset from mouse olfactory bulb tissue https://github.com/JinmiaoChenLab/
(3) 10x Visium spatial transcriptomics dataset of human breast cancer https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Breast_Cancer_Block_A_Section_1
(4) 10x Visium spatial transcriptomics dataset of Mouse brain anterior https://mouse.brain-map.org/static/atlas
(5) STARmap dataset of Mouse Visual cortex https://www.dropbox.com/sh/f7ebheru1lbz91s/AADm6D54GSEFXB1feRy6OSASa/visual_1020/20180505_BY3_1kgenes?dl=0&subfolder_nav_tracking=1.

