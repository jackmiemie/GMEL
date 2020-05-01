import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn


class GATLayer(nn.Module):
    def __init__(self, g, in_ndim, out_ndim, in_edim=1, out_edim=1):
        '''
        g: the graph
        in_dim: input node feature dimension
        out_dim: output node feature dimension
        edf_dim: input edge feature dimension
        '''
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc0 = nn.Linear(in_edim, out_edim, bias=False)
        self.fc1 = nn.Linear(in_ndim, out_ndim, bias=False)
        self.fc2 = nn.Linear(in_ndim, out_ndim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_ndim + out_edim, 1, bias=False)
        # equation (4)
        self.activation = F.relu
        # parameters
        self.weights = nn.Parameter(torch.Tensor(2, 1)) # used to calculate convex combination weights
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('relu'))

    def edge_feat_func(self, edges):
        '''
        deal with edge features
        '''
        return {'t': self.fc0(edges.data['d'])}

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z'], edges.data['t']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'],'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4). this is the core update part.
        z_neighbor = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        z_i = nodes.data['z_i']
        # calculate the convex combination weights
        lambda_ = F.softmax(self.weights, dim=0)
        # update
        h = self.activation(z_i + z_neighbor)
        return {'h': h}

    def forward(self, h):
        # equation (1)
        self.g.apply_edges(self.edge_feat_func)
        z = self.fc1(h) 
        self.g.ndata['z'] = z # message passed to the others
        z_i = self.fc2(h) 
        self.g.ndata['z_i'] = z_i # message passed to self
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')
    
class GATInputLayer(nn.Module):

    def __init__(self, g, in_ndim, out_ndim, in_edim=1, out_edim=1):
        '''
        g: the graph
        in_ndim: input node feature dimension
        out_ndim: output node feature dimension
        in_edim: input edge feature dimension
        out_edim: output edge feature dimension
        dropout: dropout rate
        '''
        # initialize super class
        super().__init__()
        # handle parameters
        self.g = g
        # equation (1)
        self.fc0 = nn.Linear(in_edim, out_edim, bias=False)
        self.fc1 = nn.Linear(in_ndim, out_ndim, bias=False)
        self.fc2 = nn.Linear(in_ndim, out_ndim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_ndim + out_edim, 1, bias=False)
        # equation (4)
        self.activation = F.relu
        # parameters
        self.weights = nn.Parameter(torch.Tensor(2, 1)) # used to calculate convex combination weights
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('relu'))

    def edge_feat_func(self, edges):
        '''
        transform edge features
        '''
        return {'t': self.fc0(edges.data['d'])}

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z'], edges.data['t']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4). this is the core update part.
        z_neighbor = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        z_i = nodes.data['z_i']
        # calculate the convex combination weights
        lambda_ = F.softmax(self.weights, dim=0)
        # update
        h = self.activation(z_i + z_neighbor)
        return {'h': h}

    def forward(self, attr):
        # equation (1)
        self.g.apply_edges(self.edge_feat_func)
        z = self.fc1(attr) # message passed to the others
        self.g.ndata['z'] = z
        z_i = self.fc2(attr) # message passed to self
        self.g.ndata['z_i'] = z_i
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')