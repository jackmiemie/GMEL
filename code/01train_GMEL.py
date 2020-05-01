import pandas as pd
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import logging 

# imports from local
import train

def main(args):
    # logger
    logger = logging.getLogger('pretraining') # experiment name
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler("log/01pre_training.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    # set random seed for reproducibility
    torch.manual_seed(2019)
    np.random.seed(2019)

    # fix args
    fix_args = {}
    fix_args['year'] = args['year']
    fix_args['device'] = args['device']
    fix_args['reg_param'] = args['reg_param']
    fix_args['max_epochs'] = args['max_epochs']
    fix_args['mini_batch_size'] = args['mini_batch_size']
    fix_args['negative_sampling_rate'] = args['negative_sampling_rate']
    fix_args['lr'] = args['lr']
    fix_args['grad_norm'] = args['grad_norm']
    fix_args['evaluate_every'] = args['evaluate_every']

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
    
    # pretraining
    for control in controls:
        # construct training arguments
        train_args = {'num_hidden_layers': control[0], 'embedding_size': control[1], 'multitask_weights': control[2]}
        train_args.update(fix_args)
        # training
        logger.info('-------------------------------------------')
        logger.info('num_hidden_layers{}, embedding_size{}, multitask_weights{}'.format(train_args['num_hidden_layers'], train_args['embedding_size'], train_args['multitask_weights']))
        train.main(train_args)
        logger.info('finish training')
        logger.info('-------------------------------------------')
    # report
    logger.info('all done')


if __name__ == "__main__":
    args = {'year': 2015, 'device': 'cuda:0',
            'reg_param': 0, 'max_epochs': 120, 'mini_batch_size': 1e3, 'negative_sampling_rate': 0,
            'lr': 1e-5, 'grad_norm': 1.0,
            'evaluate_every': 5}
    main(args)