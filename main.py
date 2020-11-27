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
    config['snapshot_dir'] = 'snapshot/%s' % config['exp_dir']
    config['tboard_dir'] = 'snapshot/%s/tensorboard' % config['exp_dir']
    config['save_dir'] = 'snapshot/%s/checkpoints' % config['exp_dir']
    
    config = edict(config)

    os.makedirs(config.snapshot_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.tboard_dir, exist_ok=True)
    json.dump(
        config,
        open(os.path.join(config.snapshot_dir, 'config.json'), 'w'),
        indent=4,
    )
    
    # backup the files
    os.system(f'cp -r models {config.snapshot_dir}')
    os.system(f'cp -r datasets {config.snapshot_dir}')
    os.system(f'cp -r lib {config.snapshot_dir}')
    shutil.copy2('main.py',config.snapshot_dir)
    
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

    
    # create optimizer 
    if config.optimizer == 'SGD':
        config.optimizer = optim.SGD(
            config.model.parameters(), 
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            )
    elif config.optimizer == 'ADAM':
        config.optimizer = optim.Adam(
            config.model.parameters(), 
            lr=config.lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )
    
    # create learning rate scheduler
    config.scheduler = optim.lr_scheduler.ExponentialLR(
        config.optimizer,
        gamma=config.scheduler_gamma,
    )
    
    # create dataset and dataloader
    info_train = load_obj(config.train_info)
    info_val = load_obj(config.val_info)
    info_benchmark = load_obj(f'configs/{config.test_info}.pkl')

    train_set = ThreeDMatchDownsampled(info_train,config,data_augmentation=True)
    val_set = ThreeDMatchDownsampled(info_val,config,data_augmentation=False)
    benchmark_set = ThreeDMatchDownsampled(info_benchmark,config, data_augmentation=False)

    config.train_loader, neighborhood_limits = get_dataloader(dataset=train_set,
                                        batch_size=config.batch_size,
                                        shuffle=True,
                                        num_workers=config.num_workers,
                                        )
    config.val_loader, _ = get_dataloader(dataset=val_set,
                                        batch_size=config.batch_size,
                                        shuffle=False,
                                        num_workers=1,
                                        neighborhood_limits=neighborhood_limits
                                        )
    config.test_loader, _ = get_dataloader(dataset=benchmark_set,
                                        batch_size=config.batch_size,
                                        shuffle=False,
                                        num_workers=1,
                                        neighborhood_limits=neighborhood_limits)
    
    # create evaluation metrics
    config.desc_loss = MetricLoss(config)

    # start to train our model
    trainer = Trainer(config)
    if(config.mode=='train'):
        trainer.train()
    elif(config.mode =='val'):
        trainer.eval()
    else:
        trainer.test()
    