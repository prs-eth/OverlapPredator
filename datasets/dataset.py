"""
Author: Shengyu Huang
Last modified: 30.11.2020
"""

import os,sys,glob,torch
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
import open3d as o3d

def get_correspondences(src_pcd, tgt_pcd, trans, search_voxel_size, K=None):
    src_pcd.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd)

    correspondences = []
    for i, point in enumerate(src_pcd.points):
        [count, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            correspondences.append([i, j])
    
    correspondences = np.array(correspondences)
    correspondences = torch.from_numpy(correspondences)
    return correspondences

def to_pcd(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def to_tsfm(rot,trans):
    tsfm = np.eye(4)
    tsfm[:3,:3]=rot
    tsfm[:3,3]=trans.flatten()
    return tsfm


class ThreeDMatchDownsampled(Dataset):
    """
    Load subsampled coordinates, relative rotation and translation
    Output(torch.Tensor):
        src_pcd:        [N,3]
        tgt_pcd:        [M,3]
        rot:            [3,3]
        trans:          [3,1]
    """
    def __init__(self,infos,config,data_augmentation=True):
        super(ThreeDMatchDownsampled,self).__init__()
        self.infos = infos
        self.base_dir = config.root
        self.pos_radius = config.pos_radius
        self.data_augmentation=data_augmentation
        self.config = config
        
        self.rot_factor=1.
        self.augment_noise = config.augment_noise
        self.max_points = 30000

    def __len__(self):
        return len(self.infos['rot'])

    def __getitem__(self,item): 
        # get transformation
        rot=self.infos['rot'][item]
        trans=self.infos['trans'][item]

        # get pointcloud
        src_path=os.path.join(self.base_dir,self.infos['src'][item])
        tgt_path=os.path.join(self.base_dir,self.infos['tgt'][item])
        src_pcd = torch.load(src_path)
        tgt_pcd = torch.load(tgt_path)

        # if we get too many points, we do some downsampling
        if(src_pcd.shape[0] > self.max_points):
            idx = np.random.permutation(src_pcd.shape[0])[:self.max_points]
            src_pcd = src_pcd[idx]
        if(tgt_pcd.shape[0] > self.max_points):
            idx = np.random.permutation(tgt_pcd.shape[0])[:self.max_points]
            tgt_pcd = tgt_pcd[idx]

        # add gaussian noise
        if self.data_augmentation:            
            # rotate the point cloud
            euler_ab=np.random.rand(3)*np.pi*2/self.rot_factor # anglez, angley, anglex
            rot_ab= Rotation.from_euler('zyx', euler_ab).as_matrix()
            if(np.random.rand(1)[0]>0.5):
                src_pcd=np.matmul(rot_ab,src_pcd.T).T
                rot=np.matmul(rot,rot_ab.T)
            else:
                tgt_pcd=np.matmul(rot_ab,tgt_pcd.T).T
                rot=np.matmul(rot_ab,rot)
                trans=np.matmul(rot_ab,trans)

            src_pcd += (np.random.rand(src_pcd.shape[0],3) - 0.5) * self.augment_noise
            tgt_pcd += (np.random.rand(tgt_pcd.shape[0],3) - 0.5) * self.augment_noise
        
        if(trans.ndim==1):
            trans=trans[:,None]

        # get correspondence at fine level
        tsfm = to_tsfm(rot, trans)
        correspondences = get_correspondences(to_pcd(src_pcd), to_pcd(tgt_pcd), tsfm,self.pos_radius)
            
        src_feats=np.ones_like(src_pcd[:,:1]).astype(np.float32)
        tgt_feats=np.ones_like(tgt_pcd[:,:1]).astype(np.float32)
        rot = rot.astype(np.float32)
        trans = trans.astype(np.float32)
        
        return src_pcd,tgt_pcd,src_feats,tgt_feats,rot,trans, correspondences