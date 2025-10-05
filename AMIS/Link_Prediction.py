import torch
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_matrix
import time

from src.Utils import load_our_data, get_model
from src.args import get_citation_args
from src.link_prediction_evaluate import predict_model

args = get_citation_args()



# IMDB
# args.dataset = 'IMDB'
# eval_name = r'data/IMDB'
# net_path = r"data/IMDB/IMDB.mat"
# savepath = r'data/IMDB'
# encode_path=r'data/IMDB/IMDB_encoding.txt'
# eval_name = r'IMDB'
# file_name = r'data/IMDB'
# eval_type = 'all'


# Alibaba
args.dataset = 'Alibaba'
eval_name = r'Alibaba'
net_path = r'data/Alibaba/Alibaba.mat'
savepath = r'data/Alibaba'
eval_name = r'Alibaba'
encode_path=r'data/Alibaba/Alibaba_encoding.txt'
file_name = r'data/Alibaba'
eval_type = 'all'

# mat = loadmat(net_path)
# 
# try:
#     train = mat['A']
# except:
#     try:
#         train = mat['train']+mat['valid']+mat['test']
#     except:
#         try:
#             train = mat['train_full']+mat['valid_full']+mat['test_full']
#         except:
#             try:
#                 train = mat['edges']
#             except:
#                 train = np.vstack((mat['edge1'],mat['edge2']))
# 
# try:
#     feature = mat['full_feature']
# except:
#     try:
#         feature = mat['feature']
#     except:
#         try:
#             feature = mat['features']
#         except:
#             feature = mat['node_feature']
# 
# feature = csc_matrix(feature) if type(feature) != csc_matrix else feature
# 
# if net_path == 'imdb_1_10.mat':
#     A = train[0]
# elif args.dataset == 'Aminer_10k_4class':
#     A = [[mat['PAP'], mat['PCP'], mat['PTP'] ]]
#     feature = mat['node_feature']
#     feature = csc_matrix(feature) if type(feature) != csc_matrix else feature
# else:
#     A = train

print('start')
# new_adj = torch.load(adj_path)
mat = loadmat(net_path)
encode=np.loadtxt(encode_path)
encode=torch.tensor(encode)
print('end')
try:
    train = mat['A']
except:
    try:
        train = mat['train']+mat['valid']+mat['test']
    except:
        try:
            train = mat['train_full']+mat['valid_full']+mat['test_full']
        except:
            try:
                train = mat['edges']
            except:
                train = mat['edge']

try:
    feature = mat['full_feature']
except:
    try:
        feature = mat['feature']
    except:
        try:
            feature = mat['features']
        except:
            feature = mat['node_feature']
A = train[0][0]
feature = csc_matrix(feature) if type(feature) != csc_matrix else feature

if net_path == 'imdb_1_10.mat':
    A = train[0]
else:
    A = train
if args.dataset == 'alibaba_small':
    mat=loadmat(net_path)
    A=mat['edge']

node_matching = False

adj, features, labels, idx_train, idx_val, idx_test = load_our_data(args.dataset, False)
model = get_model(args.model, features.size(1), labels.max().item()+1, A, args.hidden, args.out, args.dropout, False)

starttime=time.time()
ROC,  PR = predict_model(model, file_name, feature, A,encode, eval_type, node_matching)
endtime=time.time()

print('Test ROC: {:.10f},  PR: {:.10f}'.format(ROC,  PR))
