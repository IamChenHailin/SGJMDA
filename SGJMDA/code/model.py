import numpy as np
import torch.nn.functional as fun
from abc import ABC
import torch.optim as optim
import torch
#torch.backends.cudnn.enabled = False
from utils import *
from dgl import function as fn
from torch_geometric.nn import MessagePassing, JumpingKnowledge
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import softmax
from layer import *
from torch_geometric.nn import conv
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import matplotlib
matplotlib.use("TkAgg")
matplotlib.interactive(True)
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv
from keras import layers, models
import tensorflow as tf



class ConstructAdjMatrix(nn.Module, ABC):
    def __init__(self, original_adj_mat, device="cpu"):
        super(ConstructAdjMatrix, self).__init__()
        self.adj = original_adj_mat.to(device)
        self.device = device

    def forward(self):
        d_x = torch.diag(torch.pow(torch.sum(self.adj, dim=1)+1, -0.5))
        d_y = torch.diag(torch.pow(torch.sum(self.adj, dim=0)+1, -0.5))

        agg_cell_lp = torch.mm(torch.mm(d_x, self.adj), d_y)
        agg_drug_lp = torch.mm(torch.mm(d_y, self.adj.T), d_x)

        d_c = torch.pow(torch.sum(self.adj, dim=1)+1, -1)
        self_cell_lp = torch.diag(torch.add(d_c, 1))
        d_d = torch.pow(torch.sum(self.adj, dim=0)+1, -1)
        self_drug_lp = torch.diag(torch.add(d_d, 1))
        return agg_cell_lp, agg_drug_lp, self_cell_lp, self_drug_lp

class ConstructAdjMatrix1(nn.Module, ABC):
    def __init__(self, original_adj_mat, cell_feat, drug_feat, device):
        super(ConstructAdjMatrix1, self).__init__()
        self.adj = original_adj_mat.to(device)
        self.cell_feat = cell_feat
        self.drug_feat = drug_feat
        self.fc_cell = nn.Linear(self.cell_feat.shape[1], self.cell_feat.shape[0], bias=True).to(device)
        self.fc_drug = nn.Linear(self.drug_feat.shape[1], self.drug_feat.shape[0], bias=True).to(device)
        self.lc = nn.BatchNorm1d(self.cell_feat.shape[0]).to(device)
        self.ld = nn.BatchNorm1d(self.drug_feat.shape[0]).to(device)
        self.device = device
        # cell = self.lc(self.fc_cell(self.cell_feat))
        # drug = self.ld(self.fc_drug(self.drug_feat))
        # cell = torch.cat((cell, self.adj), dim=1)
        # self.cell_encoder, _ = self.miRNA_auto_encoder(cell)
        # drug = torch.cat((self.adj.T, drug), dim=1)
        # self.drug_encoder, _ = self.disease_auto_encoder(drug)


    def forward(self):
        cell = self.lc(self.fc_cell(self.cell_feat))
        drug = self.ld(self.fc_drug(self.drug_feat))
        # matrix1 = torch.zeros([drug.shape[0], cell.shape[0]])
        # matrix2 = torch.zeros([drug.shape[1], drug.shape[1]])
        # matrix3 = torch.cat((matrix1, matrix2), dim=1).to(device=self.device)
        cell = torch.cat((cell, self.adj), dim=1)
        # _, cell_encoder = self.miRNA_auto_encoder(cell)
        #cell1 = torch.cat((cell, matrix3), dim=0)
        # matrix4 = torch.zeros([self.cell_feat.shape[0], self.cell_feat.shape[0]])
        # matrix5 = torch.zeros([self.cell_feat.shape[0], drug.shape[0]])
        # matrix6 = torch.cat((matrix4, matrix5), dim=1).to(device=self.device)
        drug = torch.cat((self.adj.T, drug), dim=1)
        # _, drug_encoder = self.disease_auto_encoder(drug)
        # cell_encoder = torch.tensor(cell_encoder).to(device=self.device)
        # drug_encoder = torch.tensor(drug_encoder).to(device=self.device)
        #drug1 = torch.cat((matrix6, drug), dim=0)
        #adj = torch.cat((cell, drug), dim=0)
        # d_x1 = torch.diag(torch.pow(torch.sum(adj, dim=1)+1, -0.5))
        # d_y1 = torch.diag(torch.pow(torch.sum(adj, dim=0)+1, -0.5))
        #
        #
        #
        # agg_adj_lp = torch.mm(torch.mm(d_x1, adj), d_y1)
        d_x = torch.diag(torch.pow(torch.sum(self.adj, dim=1)+1, -0.5))
        d_y = torch.diag(torch.pow(torch.sum(self.adj, dim=0)+1, -0.5))
        # cell1 = torch.cat((d_x, self.adj), dim=1)
        # drug1 = torch.cat((self.adj.T, d_y), dim=1)
        adj = torch.cat((cell, drug), dim=0)
        d_x1 = torch.diag(torch.pow(torch.sum(adj, dim=1)+1, -0.5))
        d_y1 = torch.diag(torch.pow(torch.sum(adj, dim=0)+1, -0.5))

        agg_cell_lp = torch.mm(torch.mm(d_x, self.adj), d_y)
        agg_drug_lp = torch.mm(torch.mm(d_y, self.adj.T), d_x)
        agg_adj_lp = torch.mm(torch.mm(d_x1, adj), d_y1)

        d_c = torch.pow(torch.sum(cell, dim=1)+1, -1)
        self_cell_lp = torch.diag(torch.add(d_c, 1))
        d_d = torch.pow(torch.sum(drug, dim=1)+1, -1)
        self_drug_lp = torch.diag(torch.add(d_d, 1))
        d_e = torch.pow(torch.sum(adj, dim=1)+1, -1)
        self_adj_lp = torch.diag(torch.add(d_e, 1))


        return agg_cell_lp, agg_drug_lp, agg_adj_lp, self_cell_lp, self_drug_lp, self_adj_lp



class LoadFeature(nn.Module, ABC):
    def __init__(self, cell_exprs, drug_finger, device):
        super(LoadFeature, self).__init__()
        cell_exprs = torch.from_numpy(cell_exprs).to(device)
        self.cell_feat = torch_z_normalized(cell_exprs, dim=1).to(device)
        self.drug_feat = torch.from_numpy(drug_finger).to(device)

    def forward(self):
        #cell_feat = self.cell_feat
        #drug_feat = self.drug_feat
        return self.cell_feat, self.drug_feat

class GraphConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, adjacency_matrix, features):
        # Calculate the degree matrix
        degree_matrix = torch.sum(adjacency_matrix, dim=1, keepdim=True)

        # Normalize the adjacency matrix
        normalized_adjacency_matrix = adjacency_matrix / degree_matrix

        # Perform the graph convolution
        convoluted_features = torch.matmul(normalized_adjacency_matrix, features)
        convoluted_features = self.linear(convoluted_features)
        return convoluted_features



class GEncoder(nn.Module, ABC):
    def __init__(self, md_graph, res, cell_feat, drug_feat, layer_size,  device):
        super(GEncoder, self).__init__()
        # self.agg_cell_lp = agg_cell_lp
        # self.agg_drug_lp = agg_drug_lp
        # self.self_cell_lp = self_cell_lp
        # self.self_drug_lp = self_drug_lp
        self.md_graph = md_graph
        self.layers = layer_size
        self.cell_feat = torch.tensor(cell_feat).float().to(device)
        self.drug_feat = torch.tensor(drug_feat).float().to(device)
        # self.cell = torch.tensor(cell).float().to(device)
        # self.drug = torch.tensor(drug).float().to(device)
        self.res = res
        self.device = device
        #self.cell_edge = self.get_edge_index(self.res).to(device)
        #self.drug_edge = self.get_edge_index(self.res.T).to(device)

        self.cell_edge = self.get_edge_index(self.cell_feat).to(device)
        self.drug_edge = self.get_edge_index(self.drug_feat).to(device)
        # self.cell_edge1 = self.get_edge_index(self.cell).to(device)
        # self.drug_edge1 = self.get_edge_index(self.drug).to(device)

        #self.edge2 = self.get_edge_index(res).to(device)

        self.fc_cell = nn.Linear(self.cell_feat.shape[1], self.layers[0], bias=True).to(device)
        self.fc_drug = nn.Linear(self.drug_feat.shape[1], self.layers[0], bias=True).to(device)
        # #self.fc_cell1 = nn.Linear(1024, layer_size[0], bias=True).to(device)
        # #self.fc_drug1 = nn.Linear(1024, layer_size[0], bias=True).to(device)
        # self.lc = nn.BatchNorm1d(self.cell_feat.shape[1]).to(device)
        # self.ld = nn.BatchNorm1d(self.drug_feat.shape[1]).to(device)
        # self.lm_cell = nn.Linear(self.layers[0], self.layers[1], bias=True).to(device)
        # self.lm_drug = nn.Linear(self.layers[0], self.layers[1], bias=True).to(device)
        # #self.gat_cell = GAT(n_feat=layer_size[0],n_hid=640,n_class=int(res.max()) + 1,dropout=0.6,n_heads=8,alpha=0.2)
        # #self.gat_drug = GAT(n_feat=layer_size[0],n_hid=640,n_class=int(res.max()) + 1,dropout=0.6,n_heads=8,alpha=0.2)
        # #self.gat_cell = conv.GATConv(self.layers[0], 64)
        # #self.gat_drug = conv.GATConv(self.layers[0], 64)
        # self.gat = conv.GATConv(self.layers[0], 32).to(device)
        # #self.cell_feat = self.lc(self.fc_cell(self.cell_feat))
        # #self.drug_feat = self.ld(self.fc_drug(self.drug_feat))
        # self.att_d = Parameter(torch.ones((1, 4)), requires_grad=True).to(device)
        # self.att_m = Parameter(torch.ones((1, 4)), requires_grad=True).to(device)
        # self.conv1 = nn.Sequential(nn.Conv2d(layer_size[0], layer_size[0], 5, 1, 2), nn.ReLU())
        self.gcn_1 = GCNConv(self.cell_feat.shape[1], layer_size[1])
        self.gcn_2 = GCNConv(self.drug_feat.shape[1], layer_size[1])
        self.gcn_3 = GCNConv(self.cell_feat.shape[1], layer_size[1])
        self.gcn_4 = GCNConv(self.drug_feat.shape[1], layer_size[1])
        self.gcn_md = GraphEmbbeding(layer_size[0], layer_size[0], layer_size[0], 4, 'sum', bool,bool, 'relu', 2, 0.0)
    def forward(self):
        # self.cell_feat = (self.fc_cell(self.cell_feat))
        # self.drug_feat = (self.fc_drug(self.drug_feat))
        #
        # feature = torch.cat((cell_fc, drug_fc), dim=0)
        #
        # feature = torch.relu(self.gcn_1(feature,  self.edge))
        #
        # cell_fc = feature[:962, :]
        # drug_fc = feature[962:1190, :]





        #cell_feat = torch.tensor(cell_feat).cpu().detach().numpy()
        #drug_feat = torch.tensor(drug_feat).cpu().detach().numpy()
        #self.feature = torch.tensor(np.vstack((cell_feat, drug_feat))).to(device=self.device)
        #self.feature = torch.tensor(np.vstack((np.array(cell_feat), np.array(drug_feat)))).to(device=self.device)


        #self.cell_feat = (torch.mm(self.self_cell_lp, self.cell_feat)+torch.mm(self.agg_cell_lp, self.drug_feat))
        #self.drug_feat = (torch.mm(self.self_drug_lp, self.drug_feat)+torch.mm(self.agg_drug_lp, self.cell_feat))
        #t.relu(self.gcn_1(x, adj['edge_index']))
        cell_gcn = torch.relu(self.gcn_1(self.cell_feat, self.cell_edge, self.cell_feat[self.cell_edge[0], self.cell_edge[1]]))
        drug_gcn = torch.relu(self.gcn_2(self.drug_feat, self.drug_edge, self.drug_feat[self.drug_edge[0], self.drug_edge[1]]))
        # cell_gcn1 = torch.relu(self.gcn_3(self.cell, self.cell_edge1, self.cell[self.cell_edge1[0], self.cell_edge1[1]]))
        # drug_gcn1 = torch.relu(self.gcn_4(self.drug, self.drug_edge1, self.drug[self.drug_edge1[0], self.drug_edge1[1]]))
        # cell_emb = torch.cat((cell_gcn, cell_gcn1), dim=1)
        # drug_emb = torch.cat((drug_gcn, drug_gcn1), dim=1)
        # adj_cell_gcn = torch.relu(self.gcn_3(self.mdm[1], self.mdm[0], self.mdm[1][self.mdm[0][0], self.mdm[0][1]]))
        # adj_drug_gcn = torch.relu(self.gcn_4(self.dmd[1], self.dmd[0], self.dmd[1][self.dmd[0][0], self.dmd[0][1]]))
        # cell_emb = torch.cat((cell_gcn, adj_cell_gcn), dim=1)
        # drug_emb = torch.cat((drug_gcn, adj_drug_gcn), dim=1)
        # cell_gcn1 = torch.relu(self.gcn_3(cell_gcn, self.cell_edge, cell_gcn[self.cell_edge[0], self.cell_edge[1]]))
        # drug_gcn1 = torch.relu(self.gcn_4(drug_gcn, self.drug_edge, drug_gcn[self.drug_edge[0], self.drug_edge[1]]))
        # adj_gcn = self.gcn_md(self.md_graph, torch.cat((cell_gcn, drug_gcn), dim=0))
        # cell_emb = adj_gcn[:self.res.shape[0], :]
        # drug_emb = adj_gcn[self.res.shape[0]:, :]
        # cell_emb = torch.cat((cell_gcn, cell_emb), dim=1)
        # drug_emb = torch.cat((drug_gcn, drug_emb), dim=1)
        #self.feature = fun.relu(self.gat(self.feature, self.edge)).to(device=self.device)
        #cell_feat = self.feature[:962,:]
        #drug_feat = self.feature[962:1190,:]

        # cell_fc = cell_fc.view(cell_fc.size()[0], cell_fc.size()[1], 1, 1)
        # drug_fc = drug_fc.view(drug_fc.size()[0], drug_fc.size()[1], 1, 1)
        # cell_fc = self.conv1(cell_fc)
        # drug_fc = self.conv1(drug_fc)
        # cell_fc = cell_fc.view(cell_fc.size()[0], -1)
        # drug_fc = drug_fc.view(drug_fc.size()[0], -1)
        #
        #
        #
        #
        #
        # cell_fc = fun.relu(self.lm_cell(cell_fc))
        # drug_fc = fun.relu(self.lm_drug(drug_fc))
        return cell_gcn, drug_gcn

    def get_edge_index(self, matrix):
        edge_index = [[], []]
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
             if matrix[i, j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
        return torch.LongTensor(edge_index)





class GDecoder(nn.Module, ABC):
    def __init__(self, gamma):
        super(GDecoder, self).__init__()
        self.gamma = gamma

    def forward(self, cell_emb, drug_emb):
        Corr = torch_corr_x_y(cell_emb, drug_emb)
        output = scale_sigmoid(Corr, alpha=self.gamma)
        return output          #adj

class LinearCorrDecoder(nn.Module, ABC):
    def __init__(self, embed_dim: int, kernel_dim: int, alpha: float):
        super(LinearCorrDecoder, self).__init__()
        self.lm_x = nn.Linear(embed_dim, kernel_dim, bias=False)
        self.lm_y = nn.Linear(embed_dim, kernel_dim, bias=False)
        self.alpha = alpha

    @staticmethod
    def corr_x_y(x: torch.Tensor, y: torch.Tensor):
        assert x.size()[1] == y.size()[1], "Different size!"
        x = torch.sub(x, torch.mean(x, dim=1).view([-1, 1]))
        y = torch.sub(y, torch.mean(y, dim=1).view([-1, 1]))
        lxy = torch.mm(x, y.t())
        lxx = torch.diag(torch.mm(x, x.t()))
        lyy = torch.diag(torch.mm(y, y.t()))
        std_x_y = torch.mm(torch.sqrt(lxx).view([-1, 1]), torch.sqrt(lyy).view([1, -1]))
        corr = torch.div(lxy, std_x_y)
        return corr

    @staticmethod
    def scale_sigmoid_activation_function(x: torch.Tensor, alpha: int or float):
        assert torch.all(x.ge(-1)) and torch.all(x.le(1)), "Out of range!"
        alpha = torch.tensor(alpha, dtype=x.dtype, device=x.device)
        x = torch.sigmoid(torch.mul(alpha, x))
        max_value = torch.sigmoid(alpha)
        min_value = torch.sigmoid(-alpha)
        output = torch.div(torch.sub(x, min_value), torch.sub(max_value, min_value))
        return output

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = self.lm_x(x)
        y = self.lm_y(y)
        out = LinearCorrDecoder.corr_x_y(x=x, y=y)
        out = LinearCorrDecoder.scale_sigmoid_activation_function(x=out, alpha=self.alpha)
        return out
class ConvLayer(nn.Module):

    def __init__(self, in_feats, out_feats, k=2, method='sum', bias=True, batchnorm=False, activation='relu',
                 dropout=0.0):
        super(ConvLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.k = k + 1
        self.method = method
        self.weights = []
        for i in range(self.k):
            self.weights.append(nn.Parameter(torch.Tensor(in_feats, out_feats)))
        self.biases = None
        self.activation = None
        self.batchnorm = None
        self.dropout = None
        if bias:
            self.biases = []
            for i in range(self.k):
                self.biases.append(nn.Parameter(torch.Tensor(out_feats)))

        self.reset_parameters()

        if activation == 'relu':
            self.activation = torch.relu
        if batchnorm:
            if method == 'cat':
                self.batchnorm = nn.BatchNorm1d(out_feats * self.k)
            else:
                self.batchnorm = nn.BatchNorm1d(out_feats)
        if dropout != 0:
            self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        for i in range(self.k):
            nn.init.xavier_uniform_(self.weights[i])
            if self.biases is not None:
                nn.init.zeros_(self.biases[i])

    def forward(self, graph, feat):

        with graph.local_scope():
            degs = (graph.out_degrees().to(device='cuda:0').float().clamp(min=1))
            norm = (torch.pow(degs, -0.5))
            shp = (norm.shape + (1,) * (feat.dim() - 1))
            norm = (torch.reshape(norm, shp))
            #norm = norm.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            if self.biases is not None:
                #feat = feat.to(device=torch.device('cpu' if torch.cuda.is_available() else 'cuda:0'))
                rst = torch.matmul(feat, (self.weights[0].to(device='cuda:0'))) + (self.biases[0].to(device='cuda:0'))

            else:
                rst = torch.matmul(feat, self.weights[0])


            for i in range(1, self.k):
                feat = feat.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                feat = feat * norm
                graph.ndata['h'] = feat
                if 'e' in graph.edata.keys():
                    graph.update_all(fn.u_mul_e('h', 'e', 'm'), fn.sum('m', 'h'))
                else:
                    graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                feat = graph.ndata.pop('h')
                feat = feat * norm

                if self.method == 'sum':
                    if self.biases is not None:
                        #feat = feat.to(device=torch.device('cpu' if torch.cuda.is_available() else 'cuda:0'))
                        y = torch.matmul(feat, (self.weights[0].to(device='cuda:0'))) + (self.biases[0].to(device='cuda:0'))
                        #y = (torch.matmul(feat, self.weights[0]) + self.biases[0])
                    else:
                        y = (torch.matmul(feat, self.weights[0]))
                    rst = (rst + y)
                elif self.method == 'mean':
                    if self.biases is not None:
                        y = torch.matmul(feat, (self.weights[0].to(device='cuda:0'))) + (self.biases[0].to(device='cuda:0'))
                        #y = (torch.matmul(feat, self.weights[0]) + self.biases[0])

                    else:
                        y = (torch.matmul(feat, self.weights[0]))
                    rst = (rst + y)
                    rst = (rst / self.k)
                elif self.method == 'cat':
                    if self.biases is not None:
                        y = (torch.matmul(feat, self.weights[0]) + self.biases[0])
                    else:
                        y = (torch.matmul(feat, self.weights[0]))
                    rst = (torch.cat((rst, y), dim=1))
            rst = rst.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
            if self.batchnorm is not None:
                rst = (self.batchnorm(rst))
            if self.activation is not None:
                rst = (self.activation(rst))
            if self.dropout is not None:
                rst = (self.dropout(rst))
            #rst = rst.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            return rst
class GraphEmbbeding(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, k, method, bias, batchnorm, activation, num_layers, dropout):
        super(GraphEmbbeding, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1:
                hid_feats = out_feats
            self.layers.append(ConvLayer(in_feats, hid_feats, k, method, bias, batchnorm, activation, dropout)).to(device='cuda:0')
            if method == 'cat':
                in_feats = hid_feats * (k + 1)
            else:
                in_feats = hid_feats

    def forward(self, graph, feat):
        for i, layer in enumerate(self.layers):
            feat = (layer(graph, feat))
            feat = feat.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        return feat







class GraphAttentionLayer(MessagePassing):
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 residual: bool, dropout: float = 0.6, slope: float = 0.2, activation: nn.Module = nn.ELU()):
        super(GraphAttentionLayer, self).__init__(aggr='add', node_dim=0)
        self.in_features = in_features
        self.out_features = out_features
        self.heads = n_heads
        self.residual = residual

        self.attn_dropout = nn.Dropout(dropout)
        self.feat_dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(negative_slope=slope)
        self.activation = activation

        self.feat_lin = Linear(in_features, out_features * n_heads, bias=True, weight_initializer='glorot')
        self.attn_vec = nn.Parameter(torch.Tensor(1, n_heads, out_features))

        # use 'residual' parameters to instantiate residual structure
        if residual:
            self.proj_r = Linear(in_features, out_features, bias=False, weight_initializer='glorot')
        else:
            self.register_parameter('proj_r', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.attn_vec)

        self.feat_lin.reset_parameters()
        if self.proj_r is not None:
            self.proj_r.reset_parameters()

    def forward(self, x, edge_idx, size=None):
        # normalize input feature matrix
        x = self.feat_dropout(x)

        x_r = x_l = self.feat_lin(x).view(-1, self.heads, self.out_features)

        # calculate normal transformer components Q, K, V
        output = self.propagate(edge_index=edge_idx, x=(x_l, x_r), size=size)

        if self.proj_r is not None:
            output = (output.transpose(0, 1) + self.proj_r(x)).transpose(1, 0)

        output = self.activation(output)
        output = output.mean(dim=1)
        # output = normalize(output, p=2., dim=-1)

        return output

    def message(self, x_i, x_j, index, ptr, size_i):
        x = x_i + x_j
        x = self.leakyrelu(x)
        alpha = (x * self.attn_vec).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_dropout(alpha)

        return x_j * alpha.unsqueeze(-1)










class SGJMDA(nn.Module, ABC):
    def __init__(self, n_in_features: int, num_hidden_layer, hid_features: list, n_heads, adj_mat, cell_exprs, drug_finger, layer_size, alpha,  gamma, md_gragh, add_layer_attn, dropout,
                 device):
        super(SGJMDA, self).__init__()
        self.md_gragh = md_gragh.to(device=device)
        self.cell_feat = cell_exprs
        self.drug_feat = drug_finger
        self.num_hidden_layer = num_hidden_layer
        self.lin_m = nn.Linear(layer_size[0], layer_size[0], bias=True)
        self.lin_d = nn.Linear(layer_size[0], layer_size[0], bias=True)
        self.hid_features = hid_features
        self.encoder = GEncoder(self.md_gragh, adj_mat, self.cell_feat, self.drug_feat, layer_size,  device)
        self.decoder = GDecoder(gamma=gamma)
        # self.decoder1 = LinearCorrDecoder(1190, 192,  6.9)
        self.gcn_md = GraphEmbbeding(layer_size[0], layer_size[0], layer_size[0], 4, 'sum', bool, bool, 'relu', 2, 0.0)
        self.conv = nn.ModuleList()
        tmp = [n_in_features] + self.hid_features
        for i in range(self.num_hidden_layer):
            self.conv.append(
                GraphAttentionLayer(tmp[i], tmp[i + 1], n_heads[i], residual=False),
            )

        if n_in_features != self.hid_features[0]:
            self.proj = Linear(n_in_features, self.hid_features[0], weight_initializer='glorot', bias=True)
        else:
            self.register_parameter('proj', None)
        if add_layer_attn:
            self.JK = JumpingKnowledge('lstm', tmp[-1], self.num_hidden_layer+1)
        else:
            self.register_parameter('JK', None)

        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential()



    def forward(self,md_graph, edge_idx):
        cell_emb, drug_emb = self.encoder()
        emb_ass = self.gcn_md(md_graph, torch.cat((self.lin_m(cell_emb), self.lin_d(drug_emb)), dim=0))
        embd_list = [self.proj(emb_ass) if self.proj is not None else emb_ass]
        for i in range(2):
            emb_ass = self.conv[i](emb_ass, edge_idx)
            embd_list.append(emb_ass)

        if self.JK is not None:
            emb_ass = self.JK(embd_list)
        final_embd = self.dropout(emb_ass)

        # InnerProductDecoder
        rna_embd = final_embd[:self.cell_feat.shape[0], :]
        dis_embd = final_embd[self.cell_feat.shape[0]:, :]
        rna_embd = torch.cat((cell_emb, rna_embd), dim=1)
        dis_embd = torch.cat((drug_emb, dis_embd), dim=1)
        output = self.decoder(rna_embd, dis_embd)
        return output



class Optimizer(nn.Module, ABC):
    def __init__(self, model,edge_idx, md_graph,drug,cell,train_data, test_data, test_mask, train_mask, evaluate_fun,
                 lr=0.001, wd=1e-05, epochs=200, test_freq=20, device='cuda:0'):
        super(Optimizer, self).__init__()
        self.model = model.to(device='cuda:0')
        self.md_gragh = md_graph
        self.edge_idx =edge_idx
        self.drug = drug
        self.cell = cell
        self.train_data = train_data.to(device)
        self.test_data = test_data.to(device)
        self.test_mask = test_mask.to(device)
        self.train_mask = train_mask.to(device)
        self.evaluate_fun = evaluate_fun
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.test_freq = test_freq
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        #self.Contrastive_loss = Contrastive_loss(tau=0.5)

    def forward(self):
        true_data = torch.masked_select(self.test_data, self.test_mask)
        #best_predict = 0
        #best_auc = 0
        for epoch in torch.arange(self.epochs):
            predict_data = self.model(self.md_gragh,self.drug,self.cell,self.edge_idx)
            loss = cross_entropy_loss(self.train_data, predict_data, self.train_mask)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            predict_data_masked = torch.masked_select(predict_data, self.test_mask)
            auc = self.evaluate_fun(true_data, predict_data_masked)
            #if auc > best_auc:
                #best_auc = auc
                #best_predict = torch.masked_select(predict_data, self.test_mask)
            if epoch % self.test_freq == 0:
                print("epoch:%4d" % epoch.item(), "loss:%.6f" % loss.item(), "auc:%.4f" % auc)
        print("Fit finished.")
        return true_data, predict_data_masked
