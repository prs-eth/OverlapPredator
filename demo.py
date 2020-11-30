import os, torch, time, shutil, json,glob
import numpy as np
from config import get_config
from easydict import EasyDict as edict
from datasets.dataset import ThreeDMatchDownsampled
from datasets.dataloader import get_dataloader

from models.architectures import KPFCNN
from torch import optim
from torch import nn

from lib.utils import load_obj, setup_seed,natural_key
from lib.trainer import Trainer
from lib.loss import MetricLoss
import shutil
setup_seed(0)



if __name__ == '__main__':
    config = dict(vars((get_config())))
    
    if config.gpu_mode:
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')
    
    # model initialization
    config.architecture = [
        'simple',
        'resnetb',
    ]
    for i in range(config.num_layers-1):
        config.architecture.append('resnetb_strided')
        config.architecture.append('resnetb')
        config.architecture.append('resnetb')
    for i in range(config.num_layers-2):
        config.architecture.append('nearest_upsample')
        config.architecture.append('unary')
    config.architecture.append('nearest_upsample')
    config.architecture.append('last_unary')
    config.model = KPFCNN(config)   
    
    # create dataset and dataloader
    info_train = load_obj(config.train_info)


    info_benchmark = load_obj(f'configs/{config.test_info}.pkl')

    train_set = ThreeDMatchDownsampled(info_train,config,data_augmentation=True)
    val_set = ThreeDMatchDownsampled(info_val,config,data_augmentation=False)
    benchmark_set = ThreeDMatchDownsampled(info_benchmark,config, data_augmentation=False)

    config.train_loader, neighborhood_limits = get_dataloader(dataset=train_set,
                                        batch_size=config.batch_size,
                                        shuffle=True,
                                        num_workers=config.num_workers,
                                        )
    config.test_loader, _ = get_dataloader(dataset=benchmark_set,
                                        batch_size=config.batch_size,
                                        shuffle=False,
                                        num_workers=1,
                                        neighborhood_limits=neighborhood_limits)

    parser = argparse.ArgumentParser()
    parser.add_argument('config', default='./configs/pairwise_registration/demo/config.yaml', type=str, help='config file')
    parser.add_argument('--source_pc', default='./data/demo/pairwise/raw_data/cloud_bin_0.ply', type=str, help='source point cloud')
    parser.add_argument('--target_pc', default='./data/demo/pairwise/raw_data/cloud_bin_1.ply', type=str, help='target point cloud')
    parser.add_argument('--model', default='pairwise_reg.pt', type=str, help= 'Name of the pretrained model.')
    parser.add_argument('--verbose', action='store_true', help='Write out the intermediate results and timings')
    parser.add_argument('--visualize', action='store_true', help='Visualize the point cloud and the results.')