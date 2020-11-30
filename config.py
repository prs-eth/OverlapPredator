import argparse
import time
import os

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
  arg = parser.add_argument_group(name)
  arg_lists.append(arg)
  return arg


def str2bool(v):
  return v.lower() in ('true', '1')


# snapshot configurations
snapshot_arg = add_argument_group('Snapshot')
snapshot_arg.add_argument('--exp_dir', type=str, default='test')
snapshot_arg.add_argument('--snapshot_interval',type=int, default=1)

# KPConv Network configuration
net_arg = add_argument_group('Network')
net_arg.add_argument('--num_layers', type=int, default=4)
net_arg.add_argument('--in_points_dim', type=int, default=3)
net_arg.add_argument('--first_feats_dim', type=int, default=256) 
net_arg.add_argument('--final_feats_dim', type=int, default=32)
net_arg.add_argument('--first_subsampling_dl', type=float, default=0.025)
net_arg.add_argument('--in_features_dim', type=int, default=1)
net_arg.add_argument('--conv_radius', type=float, default=2.5)
net_arg.add_argument('--deform_radius', type=float, default=5.0)
net_arg.add_argument('--num_kernel_points', type=int, default=15)
net_arg.add_argument('--KP_extent', type=float, default=2.0)
net_arg.add_argument('--KP_influence', type=str, default='linear')
net_arg.add_argument('--aggregation_mode', type=str, default='sum', choices=['closest', 'sum'])
net_arg.add_argument('--fixed_kernel_points', type=str, default='center', choices=['center', 'verticals', 'none'])
net_arg.add_argument('--use_batch_norm', type=str2bool, default=True)
net_arg.add_argument('--batch_norm_momentum', type=float, default=0.02)
net_arg.add_argument('--deformable', type=str2bool, default=False)
net_arg.add_argument('--modulated', type=str2bool, default=False)

# GNN Network configuration
net_arg.add_argument('--gnn_feats_dim', type = int, default=512,help='feature dimention for DGCNN')
net_arg.add_argument('--dgcnn_k', type = int, default=10,help='knn graph in DGCNN')
net_arg.add_argument('--num_head', type = int, default=4,help='cross attention head')
net_arg.add_argument('--nets', type = str, default="['self','cross','self']",help='GNN configuration')

# Loss configurations
loss_arg = add_argument_group('Loss')
loss_arg.add_argument('--pos_margin', type=float, default=0.1)
loss_arg.add_argument('--neg_margin', type=float, default=1.4)
loss_arg.add_argument('--log_scale', type=float, default=24)
loss_arg.add_argument('--pos_radius', type=float,default = 0.025 * 1.5)
loss_arg.add_argument('--matchability_radius', type=float, default=0.025 * 2)
loss_arg.add_argument('--safe_radius', type =float, default=0.025 * 4)
loss_arg.add_argument('--w_circle_loss', type=float, default=1.0)
loss_arg.add_argument('--w_overlap_loss', type = float, default=1.0)
loss_arg.add_argument('--w_saliency_loss', type=float, default=0.0)
loss_arg.add_argument('--max_points', type=int, default=256,help='maximal points for circle loss')

# Optimizer configurations
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'ADAM'])
opt_arg.add_argument('--max_epoch', type=int, default=40)
opt_arg.add_argument('--lr', type=float, default=5e-3)
opt_arg.add_argument('--weight_decay', type=float, default=1e-6)
opt_arg.add_argument('--momentum', type=float, default=0.98)
opt_arg.add_argument('--scheduler', type=str, default='ExpLR')
opt_arg.add_argument('--scheduler_gamma', type=float, default=0.95)
opt_arg.add_argument('--scheduler_interval', type=int, default=1)
opt_arg.add_argument('--iter_size', type = int, default=1)


# Dataset and dataloader configurations
data_arg = add_argument_group('Data')
data_arg.add_argument('--root', type=str, default='data_ThreeDMatch')
data_arg.add_argument('--augment_noise', type=float, default=0.005)
data_arg.add_argument('--batch_size', type=int, default=1)
data_arg.add_argument('--num_workers', type=int, default=6)
data_arg.add_argument('--train_info', type = str, default='configs/train_info.pkl')
data_arg.add_argument('--val_info', type = str, default='configs/val_info.pkl')
data_arg.add_argument('--test_info', type =str, default='3DMatch')

# Demo configurations
demo_arg = add_argument_group('Demo')
demo_arg.add_argument('--source_pc', default=None, type=str, help='source point cloud')
demo_arg.add_argument('--target_pc', default=None, type=str, help='target point cloud')


# Other configurations
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--mode', type=str, default='test')
misc_arg.add_argument('--gpu_mode', type=str2bool, default=True)
misc_arg.add_argument('--verbose', type=str2bool, default=True)
misc_arg.add_argument('--verbose_freq', type = int, default=1000)
misc_arg.add_argument('--pretrain', type=str, default=None)

def get_config():
  args = parser.parse_args()
  return args
