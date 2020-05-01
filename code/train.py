import pandas as pd
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import logging 

# imports from local
import utils
from model import MyModel

def main(args):
    # device
    device = torch.device(args['device'])
    # tensorboard writer
    writer = SummaryWriter(comment='#layers{}_emb{}_multitask{}'.format(args['num_hidden_layers'], args['embedding_size'], args['multitask_weights']))
    # logger
    logger = logging.getLogger('#layers{}_emb{}_multitask{}'.format(args['num_hidden_layers'], args['embedding_size'], args['multitask_weights'])) # experiment name
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler("log/training_log.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

    # set random seed for reproducibility
    torch.manual_seed(2019)
    np.random.seed(2019)

    # load data
    data = utils.load_dataset(year=args['year'])
    # trips -- [(src, dst, cnt)]
    train_data = data['train']
    valid_data = data['valid']
    test_data = data['test']
    # in/out flow counts -- [(count)]
    train_inflow = data['train_inflow']
    train_outflow = data['train_outflow']
    # geographical features -- [[features]]
    node_feats = data['node_feats']
    # census tract adjacency as matrix
    ct_adj = data['ct_adjacency_withweight']
    # number of nodes
    num_nodes = data['num_nodes']

    # post-processing
    # trip data
    train_data = torch.from_numpy(train_data)
    trip_od_train = train_data[:, :2].long().to(device)
    trip_volume_train = train_data[:, -1].float().to(device)
    trip_od_valid = torch.from_numpy(valid_data[:, :2]).long().to(device)
    trip_volume_valid = torch.from_numpy(valid_data[:, -1]).float().to(device)
    trip_od_test = torch.from_numpy(test_data[:, :2]).long().to(device)
    trip_volume_test = torch.from_numpy(test_data[:, -1]).float().to(device)
    # in/out flow data for multitask target in/out flow
    train_inflow = torch.from_numpy(train_inflow).view(-1, 1).float().to(device)
    train_outflow = torch.from_numpy(train_outflow).view(-1, 1).float().to(device)
    # construct graph using adjacency matrix
    g = utils.build_graph_from_matrix(ct_adj, node_feats.astype(np.float32), device)
    g.to(device)

    # init model
    model = MyModel(g, num_nodes, in_dim = node_feats.shape[1], h_dim = args['embedding_size'], num_hidden_layers=args['num_hidden_layers'], dropout=0, device=device, reg_param=args['reg_param'])
    model.to(device)
    
    # training recorder
    model_state_file = './models/model_state_layers{}_emb{}_multitask{}.pth'.format(args['num_hidden_layers'], args['embedding_size'], args['multitask_weights'])
    best_rmse = 1e6

    # create a optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.1)
    
    # training process
    for epoch in range(args['max_epochs']):
        # turn model state
        model.train()
        # create a mini-batch generator
        mini_batch_gen = utils.mini_batch_gen(train_data, mini_batch_size = int(args['mini_batch_size']), num_nodes=num_nodes, negative_sampling_rate = 0)
        
        # SGD from each mini-batch
        for mini_batch in mini_batch_gen:
            # clear gradients
            optimizer.zero_grad()

            # get trip od
            trip_od = mini_batch[:, :2].long().to(device)
            # get trip volume
            scaled_trip_volume = utils.scale(mini_batch[:, -1].float()).to(device)

            # evaluate loss
            loss = model.get_loss(trip_od, scaled_trip_volume, train_inflow, train_outflow, g, multitask_weights=args['multitask_weights'])

            # add to tensorboard
            writer.add_scalar('mini_loss', loss.item(), global_step = epoch)
            
            # back propagation to get gradients
            loss.backward()
            # clip to make stable
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['grad_norm'])

            # update weights by optimizer
            optimizer.step()
        # scheduler update learning rate
        scheduler.step()
        # report training epoch
        if logger.level == logging.DEBUG:
            model.eval()
            # loss function
            with torch.no_grad():
                loss = model.get_loss(trip_od_train, utils.scale(trip_volume_train), train_inflow, train_outflow, g)
            # metric on train dataset
            rmse, mae, mape, cpc, cpl = utils.evaluate(model, g, trip_od_train, trip_volume_train)
            # report
            logger.debug("Evaluation on train dataset:")
            logger.debug("Epoch {:04d} | Loss = {:.4f}".format(epoch, loss))
            logger.debug("RMSE {:.4f} | MAE {:.4f} | MAPE {:.4f} | CPC {:.4f} | CPL {:.4f} |".format(rmse, mae, mape, cpc, cpl))
            writer.add_scalar('overall-loss', loss.item(), epoch)
            writer.add_scalar('RMSE', rmse, epoch)
            writer.add_scalar('MAE', mae, epoch)
            writer.add_scalar('MAPE', mape, epoch)
            writer.add_scalar('CPC', cpc, epoch)
            writer.add_scalar('CPL', cpl, epoch)

        # validation part
        if epoch % args['evaluate_every'] == 0:
            # turn model state to eval
            model.eval()
            # loss function
            with torch.no_grad():
                loss = model.get_loss(trip_od_valid, utils.scale(trip_volume_valid), train_inflow, train_outflow, g)
            # evaluate on validation set
            rmse, mae, mape, cpc, cpl = utils.evaluate(model, g, trip_od_valid, trip_volume_valid)
            # report
            logger.info("-----------------------------------------")
            logger.info("Evaluation on Validation:")
            logger.info("Epoch {:04d} | Loss = {:.4f}".format(epoch, loss))
            logger.info("RMSE {:.4f} | MAE {:.4f} | MAPE {:.4f} | CPC {:.4f} | CPL {:.4f} |".format(rmse, mae, mape, cpc, cpl))
            # save best model
            if rmse < best_rmse:
                # update indicator
                best_rmse = rmse
                # save model
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'rmse': rmse, 'mae': mae, 'mape': mape, 'cpc': cpc, 'cpl': cpl}, model_state_file)
                # save embeddings
                src_embedding = model(g).detach().cpu().numpy() # get embeddings
                dst_embedding = model.forward2(g).detach().cpu().numpy() # get embeddings
                emb_fp = "./embeddings/censustract_embeddings_year{}_layers{}_emb{}_multitask{}.npz".format(args['year'], args['num_hidden_layers'] , args['embedding_size'], args['multitask_weights'])
                np.savez(emb_fp, src_embedding, dst_embedding) 
                # report
                logger.info('Best RMSE found on epoch {}'.format(epoch))
            logger.info("-----------------------------------------")

if __name__ == "__main__":
    # for unit test
    args = {'year': 2015, 'device': 'cuda:0',
            'embedding_size': 128, 'num_hidden_layers': 1, 'reg_param': 0, 'multitask_weights': (0.5, 0.25, 0.25),
            'max_epochs': 100, 'mini_batch_size': 1e3, 'negative_sampling_rate': 0,
            'lr': 1e-5, 'grad_norm': 1.0,
            'evaluate_every': 5}
    main(args)