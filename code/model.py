import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn

# imports from local
from layers import GATInputLayer, GATLayer
import utils

class MyModel(nn.Module):

    def __init__(self, g, num_nodes, in_dim, h_dim, num_hidden_layers=1, dropout=0, device='cpu', reg_param=0):
        '''
        Inputs:
        ---------------------
        g: the graph
        num_nodes: number of nodes in g
        in_dim: original node attributes' dimension
        h_dim: node embedding dimension
        num_hidden_layers: number of hidden layers in graph neural network
        dropout: dropout rate
        device: device
        reg_param: regularization loss coefficient

        Output:
        ---------------------
        embedding of nodes.

        To train the model, use get_loss() to get the overall loss.
        '''
        # init super class
        super().__init__()
        # handle the parameter
        self.reg_param = reg_param
        # create modules
        self.gat = GAT(g ,num_nodes, in_dim, h_dim, h_dim, num_hidden_layers, dropout, device) # GAT for origin node
        self.gat2 = GAT(g ,num_nodes, in_dim, h_dim, h_dim, num_hidden_layers, dropout, device) # GAT for destination nodes.
        # linear plan
        self.edge_regressor = nn.Bilinear(h_dim, h_dim, 1)
        self.in_regressor = nn.Linear(h_dim, 1)
        self.out_regressor = nn.Linear(h_dim, 1)
        # FNN plan
        # self.edge_regressor = FNN(h_dim * 2, dropout, device)
        # self.in_regressor = FNN(h_dim, dropout, device)
        # self.out_regressor = FNN(h_dim, dropout, device)

    def forward(self, g):
        '''
        forward propagate of the graph to get embeddings for the origin node
        '''
        return self.gat.forward(g)
    
    def forward2(self,g):
        '''
        forward propagate of the graph to get embeddings for destination node
        '''
        return self.gat2.forward(g)

    def get_loss(self, trip_od, scaled_trip_volume, in_flows, out_flows, g, multitask_weights=[0.5, 0.25, 0.25]):
        '''
        defines the procedure of evaluating loss function

        Inputs:
        ----------------------------------
        trip_od: list of origin destination pairs
        trip_volume: ground-truth of volume of trip which serves as our target.
        g: DGL graph object

        Outputs:
        ----------------------------------
        loss: value of loss function
        '''
        # calculate the in/out flow of nodes
        # scaled back trip volume
        trip_volume = utils.scale_back(scaled_trip_volume)
        # get in/out nodes of this batch
        out_nodes, out_flows_idx = torch.unique(trip_od[:, 0], return_inverse=True)
        in_nodes, in_flows_idx = torch.unique(trip_od[:, 1], return_inverse=True)
        # scale the in/out flows of the nodes in this batch
        scaled_out_flows = utils.scale(out_flows[out_nodes])
        scaled_in_flows = utils.scale(in_flows[in_nodes])
        # get embeddings of each node from GNN
        src_embedding = self.forward(g)
        dst_embedding = self.forward2(g)
        # get edge prediction
        edge_prediction = self.predict_edge(src_embedding, dst_embedding, trip_od)
        # get in/out flow prediction
        in_flow_prediction = self.predict_inflow(dst_embedding, in_nodes)
        out_flow_prediction = self.predict_outflow(src_embedding, out_nodes)
        # get edge prediction loss
        edge_predict_loss = MSE(edge_prediction, scaled_trip_volume)
        # get in/out flow prediction loss
        in_predict_loss = MSE(in_flow_prediction, scaled_in_flows)
        out_predict_loss = MSE(out_flow_prediction, scaled_out_flows)
        # get regularization loss
        reg_loss = 0.5 * (self.regularization_loss(src_embedding) + self.regularization_loss(dst_embedding))
        # return the overall loss
        return multitask_weights[0] * edge_predict_loss + multitask_weights[1] * in_predict_loss + multitask_weights[2] * out_predict_loss + self.reg_param * reg_loss
    
    def predict_edge(self, src_embedding, dst_embedding, trip_od):
        '''
        using node embeddings to make prediction on given trip OD.
        '''
        # construct edge feature
        src_emb = src_embedding[trip_od[:,0]]
        dst_emb = dst_embedding[trip_od[:,1]]
        # get predictions
        # edge_feat = torch.cat((src_emb, dst_emb), dim=1)
        # self.edge_regressor(edge_feat)
        return self.edge_regressor(src_emb, dst_emb)

    def predict_inflow(self, embedding, in_nodes_idx):
        # make prediction
        return self.in_regressor(embedding[in_nodes_idx])

    def predict_outflow(self, embedding, out_nodes_idx):
        # make prediction
        return self.out_regressor(embedding[out_nodes_idx])

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2))


class GAT(nn.Module):

    def __init__(self, g, num_nodes, in_dim, h_dim, out_dim, num_hidden_layers=1, dropout=0, device='cpu'):
        # initialize super class
        super().__init__()
        # handle the parameters
        self.g = g
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.device = device
        # create gcn layers
        self.build_model()
    
    def build_model(self):
        self.layers = nn.ModuleList()
        # layer: input to hidden
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # layer: hidden to hidden
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # layer: hidden to output
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)
    
    def build_input_layer(self):
        act = F.relu
        return GATInputLayer(self.g, self.in_dim, self.h_dim)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return GATLayer(self.g, self.h_dim, self.h_dim)
        
    def build_output_layer(self):
        return None

    def forward(self, g):
        h = g.ndata['attr']
        for layer in self.layers:
            h = layer(h)
        return h


class Bilinear(nn.Module):

    def __init__(self, num_feats, dropout=0, device='cpu'):
        return super().__init__()
        # bilinear
        self.bilinear = nn.Bilinear(num_feats, num_feats, 1)
        # dropout
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
    
    def forward(self, x1, x2):
        return self.bilinear(x1, x2)

class FNN(nn.Module):

    def __init__(self, num_feats, dropout=0, device=False):
        # init super class
        super().__init__()
        # handle parameters
        self.in_feat = num_feats
        self.h1_feat = num_feats // 2
        self.h2_feat = self.h1_feat // 2
        self.out_feat = 1
        self.device = device
        # dropout
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        # define functions
        self.linear1 = nn.Linear(self.in_feat, self.h1_feat)
        self.linear2 = nn.Linear(self.h1_feat, self.h2_feat)
        self.linear3 = nn.Linear(self.h2_feat, self.out_feat)
        self.activation = F.relu
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        # x = F.relu(x) # enforce the prediction to be non-zero
        return x


def MSE(y_hat, y):
    '''
    Root mean square
    '''
    limit = 20000
    if y_hat.shape[0] < limit:
        return torch.mean((y_hat - y)**2)
    else:
        acc_sqe_sum = 0 # accumulative squred error sum
        for i in range(0, y_hat.shape[0], limit):
            acc_sqe_sum += torch.sum((y_hat[i: i + limit] - y[i: i + limit]) ** 2)
        return acc_sqe_sum / y_hat.shape[0]