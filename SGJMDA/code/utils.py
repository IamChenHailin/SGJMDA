import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import os
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn as nn
from scipy.sparse import coo_matrix
import math
import csv
from clac_metric import get_metrics
import matplotlib.pyplot as plt

def dir_path(k=1):
    """
    :param k: 当前路径后退级数
    :return: 后退k级后的目录
    """
    fpath = os.path.realpath(__file__)
    dir_name = os.path.dirname(fpath)
    dir_name = dir_name.replace("\\", "/")
    p = len(dir_name) - 1
    while p > 0:
        if dir_name[p] == "/":
            k -= 1
            if k == 0:
                break
        p -= 1
    p += 1
    dir_name = dir_name[0: p]
    return dir_name

def common_data_index(data_for_index: np.ndarray, data_for_cmp: np.ndarray):
    """
    :param data_for_index: data for index, numpy array
    :param data_for_cmp: data for compare, numpy array
    :return: index of common data in data for index
    """
    index = np.array([np.where(x in data_for_cmp, 1, 0) for x in data_for_index])
    index = np.where(index == 1)[0]
    return index

def to_coo_matrix(adj_mat: np.ndarray or sp.coo.coo_matrix):
    """
    :param adj_mat: adj matrix, numpy array
    :return: sparse matrix, sp.coo.coo_matrix
    """
    if not sp.isspmatrix_coo(adj_mat):
        adj_mat = sp.coo_matrix(adj_mat)
    return adj_mat

def to_tensor(positive, identity=False):
    """
    :param positive: positive sample
    :param identity: if add identity
    :return: tensor
    """
    if identity:
        data = positive + sp.identity(positive.shape[0])
    else:
        data = positive
    data = torch.from_numpy(data.toarray()).float()
    return data

def mask(positive: sp.coo.coo_matrix, negative: sp.coo.coo_matrix, dtype=int):
    """
    :param positive: positive data
    :param negative: negative data
    :param dtype: return data type
    :return: data mask
    """
    row = np.hstack((positive.row, negative.row))
    col = np.hstack((positive.col, negative.col))
    data = [1] * row.shape[0]
    masked = sp.coo_matrix((data, (row, col)), shape=positive.shape).toarray().astype(dtype)
    masked = torch.from_numpy(masked)
    return masked

def torch_z_normalized(tensor: torch.Tensor, dim=0):
    """
    :param tensor: an 2D torch tensor
    :param dim:
        0 : normalize row data
        1 : normalize col data
    :return: Gaussian normalized tensor
    """
    mean = torch.mean(tensor, dim=1-dim)
    std = torch.std(tensor, dim=1-dim)
    if dim:
        tensor_sub_mean = torch.sub(tensor, mean)
        tensor_normalized = torch.div(tensor_sub_mean, std)
    else:
        size = mean.size()[0]
        tensor_sub_mean = torch.sub(tensor, mean.view([size, -1]))
        tensor_normalized = torch.div(tensor_sub_mean, std.view([size, -1]))
    return tensor_normalized

def torch_corr_x_y(tensor1: torch.Tensor, tensor2: torch.Tensor):
    """
    :param tensor1: a matrix, torch Tensor
    :param tensor2: a matrix, torch Tensor
    :return: corr(tensor1, tensor2)
    """
    assert tensor1.size()[1] == tensor2.size()[1], "Different size!"
    tensor2 = torch.t(tensor2)
    mean1 = torch.mean(tensor1, dim=1).view([-1, 1])
    mean2 = torch.mean(tensor2, dim=0).view([1, -1])
    lxy = torch.mm(torch.sub(tensor1, mean1), torch.sub(tensor2, mean2))
    lxx = torch.diag(torch.mm(torch.sub(tensor1, mean1), torch.t(torch.sub(tensor1, mean1))))
    lyy = torch.diag(torch.mm(torch.t(torch.sub(tensor2, mean2)), torch.sub(tensor2, mean2)))
    std_x_y = torch.mm(torch.sqrt(lxx).view([-1, 1]), torch.sqrt(lyy).view([1, -1]))
    corr_x_y = torch.div(lxy, std_x_y)
    return corr_x_y

def scale_sigmoid(tensor: torch.Tensor, alpha: int or float):
    """
    :param tensor: a torch tensor, range is [-1, 1]
    :param alpha: an scale parameter to sigmod
    :return: mapping tensor to [0, 1]
    """
    alpha = torch.tensor(alpha, dtype=torch.float32, device=tensor.device)
    output = torch.sigmoid(torch.mul(alpha, tensor))
    return output

def roc_auc(true_data: torch.Tensor, predict_data: torch.Tensor):
    """
    :param true_data: train data, torch tensor 1D
    :param predict_data: predict data, torch tensor 1D
    :return: AUC score
    """
    assert torch.all(true_data.ge(0)) and torch.all(true_data.le(1)), "Out of range!"
    aucs = roc_auc_score(true_data.detach().cpu().numpy(), predict_data.detach().cpu().numpy())
    return aucs

def translate_result(tensor: torch.Tensor or np.ndarray):
    """
    :param tensor: torch tensor or np.ndarray
    :return: pd.DataFrame
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    arr = tensor.reshape((1, -1))
    arr = pd.DataFrame(arr)
    return arr

def cross_entropy_loss(true_data: torch.Tensor, predict_data:  torch.Tensor, masked: torch.Tensor):
    """
    :param true_data: true data
    :param predict_data: predict data
    :param masked: data mask
    :return: cross entropy loss
    """
    masked = masked.to(torch.bool)
    true_data = torch.masked_select(true_data, masked)
    predict_data = torch.masked_select(predict_data, masked)
    loss_fun = nn.BCELoss(reduction='mean')
    loss = loss_fun(predict_data, true_data)

    return loss

def ssl_loss(self, data1, data2):
    #index=torch.unique(index)
    embeddings1 = data1.shape[0]
    embeddings2 = data2.shape[0]
    norm_embeddings1 = torch.nn.functional.normalize(embeddings1, p = 2, dim = 1)
    norm_embeddings2 = torch.nn.functional.normalize(embeddings2, p = 2, dim = 1)
    pos_score  = torch.sum(torch.mul(norm_embeddings1, norm_embeddings2), dim = 1)
    all_score  = torch.mm(norm_embeddings1, norm_embeddings2.T)
    pos_score  = torch.exp(pos_score / self.args.ssl_temp)
    all_score  = torch.sum(torch.exp(all_score / self.args.ssl_temp), dim = 1)
    ssl_loss  = (-torch.sum(torch.log(pos_score / ((all_score))))/(len(data1.shape[0])))
    return ssl_loss

def dense2sparse(matrix: np.ndarray):
    mat_coo = coo_matrix(matrix)
    edge_idx = np.vstack((mat_coo.row, mat_coo.col))
    return edge_idx, mat_coo.data

def k_matrix(matrix, k=20):
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        knn_graph[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
        knn_graph[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]
    return knn_graph + np.eye(num)

def glorot(value):
    if isinstance(value, torch.Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)

def calculate_loss(pred, pos_edge_idx, neg_edge_idx, device):
    pos_pred_socres = pred[pos_edge_idx[0], pos_edge_idx[1]]
    neg_pred_socres = pred[neg_edge_idx[0], neg_edge_idx[1]]
    pred_scores = torch.hstack((pos_pred_socres, neg_pred_socres)).to(device)
    true_labels = torch.hstack((torch.ones(pos_pred_socres.shape[0]), torch.zeros(neg_pred_socres.shape[0]))).to(device)
    loss_fun = torch.nn.BCELoss(reduction='mean').to(device)
    # loss_fun=torch.nn.BCEWithLogitsLoss(reduction='mean')
    return loss_fun(pred_scores, true_labels).to(device)

def calculate_evaluation_metrics(pred_mat, pos_edges, neg_edges, i):
    pos_pred_socres = pred_mat[pos_edges[0], pos_edges[1]]
    neg_pred_socres = pred_mat[neg_edges[0], neg_edges[1]]
    pred_labels = np.hstack((pos_pred_socres, neg_pred_socres))
    true_labels = np.hstack((np.ones(pos_pred_socres.shape[0]), np.zeros(neg_pred_socres.shape[0])))
    # np.savetxt(f'pred_labels{i}.txt', pred_labels)
    # np.savetxt(f'true_labels{i}.txt', true_labels)
    return get_metrics1(true_labels, pred_labels)

def get_metrics1(real_score, predict_score):
    real_score, predict_score = real_score.flatten(), predict_score.flatten()
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]
    # thresholds = sorted_predict_score[range(
    #     sorted_predict_score_num )]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]

    # np.savetxt(roc_path.format(i), ROC_dot_matrix)

    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])

    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]

    # np.savetxt(pr_path.format(i), PR_dot_matrix)

    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])

    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)
    # plt.plot(x_ROC, y_ROC)
    # plt.plot(x_PR, y_PR)
    # plt.show()
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    print( ' auc:{:.4f} ,aupr:{:.4f},f1_score:{:.4f}, accuracy:{:.4f}, recall:{:.4f}, specificity:{:.4f}, precision:{:.4f}'.format(auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision))
    return [auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision]

def getGipKernel(y, trans, gamma, normalized=False):
    if trans:
        y = y.T
    if normalized:
        y = normalized_embedding(y)
    krnl = torch.mm(y, y.T)
    krnl = krnl / torch.mean(torch.diag(krnl))
    krnl = torch.exp(-kernelToDistance(krnl) * gamma)
    return krnl

def normalized_embedding(embeddings):
    [row, col] = embeddings.size()
    ne = torch.zeros([row, col])
    for i in range(row):
        ne[i, :] = (embeddings[i, :] - min(embeddings[i, :])) / (max(embeddings[i, :]) - min(embeddings[i, :]))
    return ne

def kernelToDistance(k):
    di = torch.diag(k).T
    d = di.repeat(len(k)).reshape(len(k), len(k)).T + di.repeat(len(k)).reshape(len(k), len(k)) - 2 * k
    return d

def normalized_kernel(K):
    K = abs(K)
    k = K.flatten().sort()[0]
    min_v = k[torch.nonzero(k, as_tuple=False)[0]]
    K[torch.where(K == 0)] = min_v
    D = torch.diag(K)
    D = D.sqrt()
    S = K / (D * D.T)
    return S

def constructNet(drug_dis_matrix):
    drug_matrix = np.matrix(
        np.zeros((drug_dis_matrix.shape[0], drug_dis_matrix.shape[0]), dtype=np.int8))
    dis_matrix = np.matrix(
        np.zeros((drug_dis_matrix.shape[1], drug_dis_matrix.shape[1]), dtype=np.int8))

    mat1 = np.hstack((drug_matrix, drug_dis_matrix))
    mat2 = np.hstack((drug_dis_matrix.T, dis_matrix))
    adj = np.vstack((mat1, mat2))
    return adj

def constructHNet(drug_dis_matrix, drug_matrix, dis_matrix):

    mat1 = np.hstack((drug_matrix, drug_dis_matrix))
    mat2 = np.hstack((drug_dis_matrix.T, dis_matrix))
    return np.vstack((mat1, mat2))


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()



def mp_data(adj):
    """
    construct meta-path dmd, mdm
    dmd: dm * md
    mdm: md * dm
    """
    dm = adj.transpose()
    mdm = np.matmul(adj, dm)
    mdm = sp.coo_matrix(mdm)
    # dmd.data = np.ones(dmd.data.shape[0])


    dmd = np.matmul(dm, adj)
    dmd = sp.coo_matrix(dmd)
    # mdm.data = np.ones(mdm.data.shape[0])

    #path_mdm = os.path.join(parent_dir, "mdm.npz")
    #sp.save_npz(path_mdm, mdm1)

    return mdm, dmd

def laplacian(kernel):
    d1 = sum(kernel) #tensor 1373的一维列表 每一列求和
    D_1 = torch.diag(d1) #求对角阵 得到度矩阵D 1373*1373 只有对角线上有值
    L_D_1 = D_1 - kernel #Δd
    D_5 = D_1.rsqrt() #每个元素取平方根再取倒数 因为0取不了平方根 所以为inf值
    D_5 = torch.where(torch.isinf(D_5), torch.full_like(D_5, 0), D_5) #如果值为inf则将其变成0
    L_D_11 = torch.mm(D_5, L_D_1) #这里开始是Ld的计算
    L_D_11 = torch.mm(L_D_11, D_5)
    return L_D_11

class mlp(torch.nn.Module):
    def __init__(self, num_in, num_hid1, num_hid2, num_out):
        super(mlp, self).__init__()

        self.l1 = torch.nn.Linear(num_in, num_hid1)
        self.l2 = torch.nn.Linear(num_hid1, num_hid2)
        self.classify = torch.nn.Linear(num_hid2, num_out)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.classify(x)
        x = self.sigmoid(x)
        return x

def torch_sparse_eye(num_nodes):
    indices = torch.arange(num_nodes).repeat(2, 1)
    values = torch.ones(num_nodes)
    return torch.sparse.FloatTensor(indices, values)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return np.array(md_data)