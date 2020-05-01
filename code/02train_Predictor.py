import pandas as pd
import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor

import seaborn as sns
import matplotlib.pylab as plt
import logging
import csv
import pickle

# imports from local
import utils

# set random seed for reproducibility
torch.manual_seed(2019)
np.random.seed(2019)


def construct_feat_from_srcemb_dstemb_dist(triplets, src_emb, dst_emb, dist):
    feat_src = src_emb[triplets[:, 0]]
    feat_dst = dst_emb[triplets[:, 1]]
    feat_dist = dist[triplets[:, 0], triplets[:, 1]].reshape(-1, 1)
    X = np.concatenate([feat_src, feat_dst, feat_dist], axis=1)
    y = triplets[:, 2]
    return X, y

def RMSE(y_hat, y):
    '''
    Root Mean Square Error Metric
    '''
    return np.sqrt(np.mean((y_hat - y)**2))

def CPC(y_hat, y):
    '''
    Common Part of Commuters Metric
    '''
    common = np.min(np.stack((y_hat, y), axis=1), axis=1)
    return 2 * np.sum(common) / (np.sum(y_hat) + np.sum(y))

def CPL(y_hat, y):
    '''
    Common Part of Links Metric. 
    
    Check the topology.
    '''
    yy_hat = y_hat > 0
    yy = y > 0
    return 2 * np.sum(yy_hat * yy) / (np.sum(yy_hat) + np.sum(yy))

def MAPE(y_hat, y):
    '''
    Mean Absolute Percentage Error Metric
    '''
    abserror = np.abs(y_hat - y)
    return np.mean(abserror / y)

def MAE(y_hat, y):
    '''
    Mean Absolute Error Metric
    '''
    abserror = np.abs(y_hat - y)
    return np.mean(abserror)

def evaluate(y_hat, y):
    '''
    Evaluate the error in different metrics
    '''
    # metric
    rmse = RMSE(y_hat, y)
    mae = MAE(y_hat, y)
    mape = MAPE(y_hat, y)
    cpc = CPC(y_hat, y)
    cpl = CPL(y_hat, y)
    # return
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'CPC': cpc, 'CPL': cpl}

def main(args):
    # logger
    logger = logging.getLogger('gbrt') # experiment name
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler("log/02gbrt_training.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

    # load dataset
    data = utils.load_dataset(year=args['year'])
    # parse dataset
    train_data = data['train']
    valid_data = data['valid']
    test_data = data['test']
    distm = data['distm']

    # control args -- num_hidden_layers, embedding_size, multitask_weights
    controls = [[1, 128, (0.5, 0.25, 0.25)], # change num_hidden_layers
                [0, 128, (0.5, 0.25, 0.25)],
                [2, 128, (0.5, 0.25, 0.25)],
                [3, 128, (0.5, 0.25, 0.25)],
                [1, 32, (0.5, 0.25, 0.25)], # change embedding_size
                [1, 64, (0.5, 0.25, 0.25)],
                [1, 256, (0.5, 0.25, 0.25)],
                [1, 128, (0.33, 0.33, 0.33)], # change multitask_weights
                [1, 128, (0.66, 0.165, 0.165)],
                [1, 128, (0.75, 0.125, 0.125)],
                [1, 128, (1, 0, 0)]]

    for control in controls:
        # parse the control
        num_hidden_layers = control[0]
        embedding_size = control[1]
        multitask_weights = control[2]
        # log
        logger.info('------------------------------------------------')
        logger.info('start year{} layers{} emb{} multitask{}'.format(args['year'], num_hidden_layers, embedding_size, multitask_weights))
        
        # load embeddings
        epath = './embeddings/censustract_embeddings_year{}_layers{}_emb{}_multitask{}.npz'.format(args['year'], num_hidden_layers, embedding_size, multitask_weights)
        embeddings = np.load(epath)
        # parse embeddings
        src_emb, dst_emb = embeddings['arr_0'], embeddings['arr_1']
        # scale distance matrix
        scaled_distm = distm / distm.max() * np.max([src_emb.max(), dst_emb.max()])
        # construct sample features
        X_train, y_train = construct_feat_from_srcemb_dstemb_dist(train_data, src_emb, dst_emb, scaled_distm)
        X_valid, y_valid = construct_feat_from_srcemb_dstemb_dist(valid_data, src_emb, dst_emb, scaled_distm)
        X_test, y_test = construct_feat_from_srcemb_dstemb_dist(test_data, src_emb, dst_emb, scaled_distm)
        
        # train gbrt
        logger.info('training')
        gbrt = GradientBoostingRegressor(max_depth=2, random_state=2019, n_estimators=100)
        gbrt.fit(X_train, y_train)
        logger.info('finish training')
        # test
        y_gbrt = gbrt.predict(X_test)
        res = evaluate(y_gbrt, y_test)
        logger.info('test result: {}'.format(res))
        # save result
        with open('./outputs/overall_performance.csv', 'a+') as fout:
            # create writer
            cols = ['model','RMSE', 'MAE', 'MAPE', 'CPC', 'CPL']
            writer = csv.DictWriter(fout, fieldnames=cols)
            # create model name
            model_name = 'GMEL_year{}_layers{}_emb{}_multitask{}'.format(args['year'], num_hidden_layers, embedding_size, multitask_weights)
            res['model'] = model_name
            # write result
            writer.writerow(res)
        # save model
        with open('./models/gbrt_year{}_layers{}_emb{}_multitask{}.txt'.format(args['year'], num_hidden_layers, embedding_size, multitask_weights), 'wb') as fout:
            pickle.dump(gbrt, fout)
        logger.info('------------------------------------------------')

    # report
    logger.info('all done')

if __name__ == "__main__":
    args = {'year':2015}
    main(args)