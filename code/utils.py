import pandas as pd
import numpy as np
import torch

import dgl

import numpy_indexed as npi

import warnings

def load_dataset(year=2015, dirpath='../data/LODES/', fprefix='CommutingFlow_', mappath='../data/CensusTract2010/mapping_NodeID2BoroCT2010.csv', featdir='../data/PLUTO/', featprefix='census_tract_attributes_from_pluto', adjpath = '../data/CensusTract2010/adjacency_matrix_withweight.csv'):
    '''
    Load train, valid and test dataset in a specified year.

    Returns:
    -----------------------------------
    (train_data, valid_data, test_data): a tuple of train, valid, test dataset. Each dataset is represented by a numpy array where each row is a trip sample (src, dst, count).
    '''
    # load train data
    train = pd.read_csv(dirpath + fprefix + str(year) + '_train.csv')
    # load valid data
    valid = pd.read_csv(dirpath + fprefix + str(year) + '_valid.csv')
    # load test data
    test = pd.read_csv(dirpath + fprefix + str(year) + '_test.csv')
    # mapping BoroCT2010 to node id
    mapping_table = pd.read_csv(mappath)
    train = ct2nid(train, mapping_table)
    valid = ct2nid(valid, mapping_table)
    test = ct2nid(test, mapping_table)
    # construct in/out flow count for training
    inflow_train = pd.DataFrame(index=mapping_table['node_id']) # container
    outflow_train = pd.DataFrame(index=mapping_table['node_id']) # container
    inflow = train.groupby('dst').agg({'count': 'sum'}) # stats
    outflow = train.groupby('src').agg({'count': 'sum'}) # stats
    inflow.index.name = 'node_id'
    outflow.index.name = 'node_id'
    inflow_train['count'] = inflow # pass the value
    outflow_train['count'] = outflow # pass the value
    inflow_train = inflow_train.fillna(0).sort_index() # fillna
    outflow_train = outflow_train.fillna(0).sort_index() # fillna
    # load node feature table
    node_feats = pd.read_csv(featdir + featprefix + str(year) + '.csv')
    node_feats['BoroCT2010'] = mapping_table.set_index('BoroCT2010').loc[node_feats['BoroCT2010']].values # map census tract to node id
    node_feats = node_feats.rename(columns={'BoroCT2010': 'nid'}).set_index('nid').sort_index()
    # sanity check
    if node_feats.isnull().values.any(): # if there is any NaN in nodes' feature table
        node_feats.fillna(0, inplace=True)
        warnings.warn('Feature table contains NaN. 0 is used to fill these NaNs')
    # normalization
    node_feats = (node_feats - node_feats.mean()) / node_feats.std()
    # load adjacency matrix
    ct_adj = pd.read_csv(adjpath, index_col=0)
    ct_inorder = mapping_table.sort_values(by='node_id')['BoroCT2010']
    ct_adj = ct_adj.loc[ct_inorder, ct_inorder.astype(str)]
    # min-max scale the weights
    ct_adj = ct_adj / ct_adj.max().max() # min is 0
    # fill nan with 0
    ct_adj = ct_adj.fillna(0)
    # load distance matrix
    distm = pd.read_csv('../data/OSRM/census_tract_trip_duration_matrix_bycar.csv', index_col='BoroCT2010')
    # define column order
    cols = ['src', 'dst', 'count']
    # return
    data = {}
    data['train'] = train[cols].values
    data['valid'] = valid[cols].values
    data['test'] = test[cols].values
    data['train_inflow'] = inflow_train.values
    data['train_outflow'] = outflow_train.values
    data['num_nodes'] = ct_adj.shape[0]
    data['node_feats'] = node_feats.values
    data['ct_adjacency_withweight'] = ct_adj.values
    data['distm'] = distm.values
    return data

def ct2nid(dataframe, mapping_table):
    '''
    Mapping BoroCT2010 code to node id.

    Inputs: 
    -----------------------------------
    dataframe: The dataframe of trips. it is supposed to have 3 columns: h_BoroCT2010, w_BoroCT2010, count
    mapping_table: The table of mapping between node id and BoroCT2010. The table is supposed to have two columns: node_id, BoroCT2010

    Returns:
    -----------------------------------
    frame: it is supposed to have 3 columns: src, dst, count. src and dst are the node id of source and target respectively.
    '''
    frame = dataframe.copy()
    mapping = mapping_table.copy()
    # do the mapping
    mapping = mapping.set_index('BoroCT2010')
    frame['src'] = mapping.loc[frame['h_BoroCT2010']].values
    frame['dst'] = mapping.loc[frame['w_BoroCT2010']].values
    # return
    return frame[['src', 'dst', 'count']]


def build_graph_from_matrix(adjm, node_feats, device='cpu'):
    '''
    Build graph using DGL library from adjacency matrix.

    Inputs:
    -----------------------------------
    adjm: graph adjacency matrix of which entries are either 0 or 1.
    node_feats: node features

    Returns:
    -----------------------------------
    g: DGL graph object
    '''
    # get edge nodes' tuples [(src, dst)]
    dst, src = adjm.nonzero()
    # get edge weights
    d = adjm[adjm.nonzero()]
    # create a graph
    g = dgl.DGLGraph()
    # add nodes
    g.add_nodes(adjm.shape[0])
    # add edges and edge weights
    g.add_edges(src, dst, {'d': torch.tensor(d).float().view(-1, 1)})
    # add node attribute, i.e. the geographical features of census tract
    g.ndata['attr'] = torch.from_numpy(node_feats).to(device)
    # compute the degree norm
    norm = comp_deg_norm(g)
    # add nodes norm
    g.ndata['norm'] = torch.from_numpy(norm).view(-1,1).to(device) # column vector
    # return
    return g

def comp_deg_norm(g):
    '''
    compute the degree normalization factor which is 1/in_degree
    '''
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm

def mini_batch_gen(train_data, mini_batch_size, num_nodes, negative_sampling_rate = 0):
    '''
    generator of mini-batch samples
    '''
    # positive data
    pos_samples = train_data
    # negative sampling to get negative data
    neg_samples = negative_sampling(pos_samples, num_nodes, negative_sampling_rate)
    # binding together
    if neg_samples is not None:
        samples = torch.cat((pos_samples, neg_samples), dim=0)
    else:
        samples = pos_samples
    # shuffle
    samples = samples[torch.randperm(samples.shape[0])]
    # cut to mini-batches and wrap them by a generator
    for i in range(0, samples.shape[0], mini_batch_size):
        yield samples[i:i+mini_batch_size]

def negative_sampling(pos_samples, num_nodes, negative_sampling_rate = 0):
    '''
    perform negative sampling by perturbing the positive samples
    '''
    # if do not require negative sampling
    if negative_sampling_rate == 0:
        return None
    # else, let's do negative sampling
    # number of negative samples
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_sampling_rate
    # create container for negative samples
    neg_samples = np.tile(pos_samples, [negative_sampling_rate, 1])
    neg_samples[:, -1] = 0 # set trip volume to be 0
    # perturbing the edge
    sample_nid = np.random.randint(num_nodes, size = num_to_generate) # randomly sample nodes
    pos_choices = np.random.uniform(size = num_to_generate) # randomly sample position
    subj = pos_choices > 0.5
    obj = pos_choices <= 0.5
    neg_samples[subj, 0] = sample_nid[subj]
    neg_samples[obj, 1] = sample_nid[obj]
    # sanity check
    while(True):
        # check overlap edges
        overlap = npi.contains(pos_samples[:, :2], neg_samples[:, :2]) # True means overlap
        if overlap.any(): # if there is any overlap edge, resample for these edges
            # get the overlap subset
            neg_samples_overlap = neg_samples[overlap]
            # resample
            sample_nid = np.random.randint(num_nodes, size = overlap.sum())
            pos_choices = np.random.uniform(size = overlap.sum())
            subj = pos_choices > 0.5
            obj = pos_choices <= 0.5
            neg_samples_overlap[subj, 0] = sample_nid[subj]
            neg_samples_overlap[obj, 1] = sample_nid[obj]
            # reassign the subset
            neg_samples[overlap] = neg_samples_overlap
        else: # if no overlap, just break resample loop
            break
    # return negative samples
    return torch.from_numpy(neg_samples)

def evaluate(model, g, trip_od, trip_volume):
    '''
    evaluate trained model.
    '''
    with torch.no_grad():
        # get embedding
        src_embedding = model(g)
        dst_embedding = model.forward2(g)
        # get prediction
        scaled_prediction = model.predict_edge(src_embedding, dst_embedding, trip_od)
        # transform back to the original scale
        prediction = scale_back(scaled_prediction)
        # get ground-truth label
        y = trip_volume.float().view(-1, 1)
        # get metric
        rmse = RMSE(prediction, y)
        mae = MAE(prediction, y)
        mape = MAPE(prediction, y)
        cpc = CPC(prediction, y)
        cpl = CPL(prediction, y)
    # return
    return rmse.item(), mae.item(), mape.item(), cpc.item(), cpl.item()

def scale(y):
    '''
    scale the target variable
    '''
    return torch.sqrt(y)

def scale_back(scaled_y):
    '''
    scale back the target varibale to normal scale
    '''
    return scaled_y ** 2

def RMSE(y_hat, y):
    '''
    Root Mean Square Error Metric
    '''
    return torch.sqrt(torch.mean((y_hat - y)**2))

def MAE(y_hat, y):
    '''
    Mean Absolute Error Metric
    '''
    abserror = torch.abs(y_hat - y)
    return torch.mean(abserror)

def MAPE(y_hat, y):
    '''
    Mean Absolute Error Metric
    '''
    abserror = torch.abs(y_hat - y)
    return torch.mean(abserror / y)

def CPC(y_hat, y):
    '''
    Common Part of Commuters Metric
    '''
    return 2 * torch.sum(torch.min(y_hat, y)) / (torch.sum(y_hat) + torch.sum(y))

def CPL(y_hat, y):
    '''
    Common Part of Links Metric. 
    
    Check the topology.
    '''
    yy_hat = y_hat > 0
    yy = y > 0
    return 2 * torch.sum(yy_hat * yy) / (torch.sum(yy_hat) + torch.sum(yy))