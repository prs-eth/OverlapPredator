"""
We use this script to calculate the overlap ratios for all the train/test fragment pairs
"""
import os,sys,glob
import open3d as o3d
from lib.utils import natural_key
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

def determine_epsilon():
    """
    We follow Learning Compact Geomtric Features to compute this hyperparameter, which unfortunately we didn't use later.
    """
    base_dir='../dataset/3DMatch/test/*/03_Transformed/*.ply'
    files=sorted(glob.glob(base_dir),key=natural_key)
    etas=[]
    for eachfile in files:
        pcd=o3d.io.read_point_cloud(eachfile)
        pcd=pcd.voxel_down_sample(0.025)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        distances=[]
        for i, point in enumerate(pcd.points):
            [count,vec1, vec2] = pcd_tree.search_knn_vector_3d(point,2)
            distances.append(np.sqrt(vec2[1]))
        etai=np.median(distances)
        etas.append(etai)
    return np.median(etas)


def get_overlap_ratio(source,target,threshold=0.03):
    """
    We compute overlap ratio from source point cloud to target point cloud
    """
    pcd_tree = o3d.geometry.KDTreeFlann(target)
    
    match_count=0
    for i, point in enumerate(source.points):
        [count, _, _] = pcd_tree.search_radius_vector_3d(point, threshold)
        if(count!=0):
            match_count+=1

    overlap_ratio = match_count / len(source.points)
    return overlap_ratio

def cal_overlap_per_scene(c_folder):
    base_dir=os.path.join(c_folder,'03_Transformed')
    fragments=sorted(glob.glob(base_dir+'/*.ply'),key=natural_key)
    n_fragments=len(fragments)

    with open(f'{c_folder}/overlaps_ours.txt','w') as f:
        for i in tqdm(range(n_fragments-1)):
            for j in range(i+1,n_fragments):
                path1,path2=fragments[i],fragments[j]
                
                # load, downsample and transform
                pcd1=o3d.io.read_point_cloud(path1)
                pcd2=o3d.io.read_point_cloud(path2)
                pcd1=pcd1.voxel_down_sample(0.01)
                pcd2=pcd2.voxel_down_sample(0.01)

                # calculate overlap
                c_overlap = get_overlap_ratio(pcd1,pcd2)
                f.write(f'{i},{j},{c_overlap:.4f}\n')
        f.close()      

if __name__=='__main__':
    base_dir='your data folder'
    scenes = sorted(glob.glob(base_dir))

    p = mp.Pool(processes=mp.cpu_count())
    p.map(cal_overlap_mat,scenes)
    p.close()
    p.join()