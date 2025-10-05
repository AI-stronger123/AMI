# AMI

This repository contains the source code and datasets for Adaptive Multi-interaction Fusion for Multi-relational Heterogeneous Networks

## Dependencies

The code is written in Python 3.9.  Before running, you need to first install the required packages :

torch==1.9.1

scipy==1.13.1

scikit-learn==1.6.1

pandas==0.25.0

## Datasets

 The datasets link as followed:

- IMDB https://github.com/RuixZh/SR-RSC
- Alibaba https://github.com/xuehansheng/DualHGCN
- Alibaba-s https://tianchi.aliyun.com/competition/entrance/231719/information/
- DBLP https://www.dropbox.com/s/yh4grpeks87ugr2/DBLP_processed.zip?dl=0
- Douban https://github.com/7thsword/MFPR-Datasets

## Preprocessing

The dataset is provided as a compressed .mat file containing the following key structures:

`edge`: This array holds the coupled subnetworks, with each element representing a distinct subnetwork.

`feature`: Contains the attributes for each node within the network.

**`train` / `valid` / `test`**: These arrays store the indices that delineate the data points for the training, validation, and testing sets required for the link prediction task.

In addition, we sample the positive and negative edges in the network, and divide them into three text files: train, valid and test. And `(dataset)_encoding.txt`is  basic behavior pattern matrice.

## Usage

First, you should modify the dataset path in the Link_prediction.py.

Second, the parameters which are in the `Model.py` and Decoupling_matrix_aggregation.py should be modified based on the quantity of actual relations and basic behavior patterns in the dataset. 


Finally, you need to execute the following command to run the link prediction task: python Link_Prediction.py
