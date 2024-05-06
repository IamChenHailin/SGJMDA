import argparse
import numpy as np
from utils import *
from sklearn.model_selection import KFold
#from model import *
import dgl
from model import *
import copy
import torch
import random
from clac_metric import get_metrics
from similarity_fusion import *
import gc
#torch.backends.cudnn.enable = True
#torch.backends.cudnn.benchmark = True
import os
torch.autograd.set_detect_anomaly(True)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#os.add_dll_directory('D:\anaconda\envs\ss\lib\site-packages\dgl\dgl.dll')
torch.cuda.empty_cache()

def set_seed(seed):
    torch.manual_seed(seed)
    #进行随机搜索的这个要注释掉
    # random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

set_seed(1)
repeat_times = 1

parser = argparse.ArgumentParser(description="Run SGJMDA")
parser.add_argument('-device', type=str, default="cuda:0", help='cuda:number or cpu')
parser.add_argument('--lr', type=float, default=0.001,
                    help="the learning rate")
parser.add_argument('--wd', type=float, default=1e-5,
                    help="the weight decay for l2 normalizaton")
parser.add_argument('--layer_size', nargs='?', default=[128, 128],
                    help='Output sizes of every layer')
parser.add_argument('--alpha', type=float, default=0.28,
                    help="the scale for balance gcn and ni")
parser.add_argument('--gamma', type=float, default=10,
                    help="the scale for sigmod")
parser.add_argument('--epochs', type=float, default=2000,
                    help="the epochs for model")

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
print(args.cuda)
hyperparam_dict = {
    'kfolds': 10,
    'num_heads_per_layer': 4,
    'num_embedding_features': 128,
    'num_hidden_layers': 2,
}



#load data
# res = pd.read_csv("../datasets/m_d.csv", index_col=0, dtype=np.float32).to_numpy()
# res = read_csv("../Data/MD_A.csv")
MD = pd.read_csv("../Data/MD_A.csv", index_col=0)
MD_c = MD.copy()
MD_c.columns = range(0, MD.shape[1])
MD_c.index = range(0, MD.shape[0])
res = np.array(MD_c)
k1 = 117
k2 = 13
m_fusion_sim, d_fusion_sim = get_fusion_sim(k1, k2)
m_fusion_sim = np.array(m_fusion_sim)
d_fusion_sim = np.array(d_fusion_sim)
pos_edges = np.loadtxt('pos.txt').astype(int)
neg_edges = np.loadtxt('neg.txt').astype(int)
idx = np.arange(pos_edges.shape[1])
np.random.shuffle(idx)
metrics_tensor = np.zeros((1, 7))
edge_idx_device = torch.tensor(np.where(res == 1), dtype=torch.long, device=args.device)
num_hidden_layers = hyperparam_dict['num_hidden_layers']
num_embedding_features = [hyperparam_dict['num_embedding_features'] for _ in range(num_hidden_layers)]
num_heads_per_layer = [hyperparam_dict['num_heads_per_layer'] for _ in range(num_hidden_layers)]
kfolds = hyperparam_dict['kfolds']
idx_splited = np.array_split(idx, kfolds)



for i in range(kfolds):
    tmp = []
    for j in range(1, kfolds):
        tmp.append(idx_splited[(j + i) % kfolds])
    training_pos_edges = np.loadtxt(f'trainpos{i}.txt').astype(int)
    training_neg_edges = np.loadtxt(f'trainneg{i}.txt').astype(int)
    test_pos_edges = np.loadtxt(f'testpos{i}.txt').astype(int)
    test_neg_edges = np.loadtxt(f'testneg{i}.txt').astype(int)
    temp_drug_dis = np.zeros((res.shape[0], res.shape[1]))
    temp_drug_dis[training_pos_edges[0], training_pos_edges[1]] = 1
    print(f'################Fold {i + 1} of {kfolds}################')
    print(i, ':after:', np.sum(temp_drug_dis.reshape(-1)))
    temp_drug_dis = torch.from_numpy(temp_drug_dis).to(torch.float32).to(device=args.device)
    md_copy = copy.deepcopy(temp_drug_dis)
    md_copy = np.array(md_copy.cpu())
    md_copy[:, 1] = md_copy[:, 1] + res.shape[0]
    md_graph = dgl.graph(
        (np.concatenate((md_copy[:, 0], md_copy[:, 1])), np.concatenate((md_copy[:, 1], md_copy[:, 0]))),
        num_nodes=res.shape[0] + res.shape[1])
    md_graph = md_graph.to(device=args.device)
    model = SGJMDA(n_in_features=args.layer_size[0], num_hidden_layer=num_hidden_layers, hid_features=num_embedding_features, n_heads=num_heads_per_layer, adj_mat=temp_drug_dis, cell_exprs=m_fusion_sim, drug_finger=d_fusion_sim, layer_size=args.layer_size, alpha=args.alpha, gamma=args.gamma, md_gragh=md_graph, add_layer_attn=True, dropout=0.6, device=args.device).to(device=args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    for epoch in range(args.epochs):
        model.train()
        out = model(md_graph, edge_idx_device)
        loss = calculate_loss(out, training_pos_edges, training_neg_edges, device=args.device)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if (epoch + 1) % 100 == 0 or epoch == 0:
            pass
            print('------EPOCH {} of {}------'.format(epoch + 1, args.epochs))
            print('Loss: {}'.format(loss))

    model.eval()
    with torch.no_grad():
        pred_mat = model(md_graph, edge_idx_device)
        pred_mat = pred_mat.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        metrics = calculate_evaluation_metrics(np.array(pred_mat.cpu()), test_pos_edges, test_neg_edges, i)
        metrics_tensor += metrics
        del temp_drug_dis
print('Average result:', end='')
avg_metrics = metrics_tensor / kfolds
del metrics_tensor
print(avg_metrics)