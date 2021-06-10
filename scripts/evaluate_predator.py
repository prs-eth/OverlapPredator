"""
Scripts for pairwise registration with RANSAC and our probabilistic sampling

Author: Shengyu Huang
Last modified: 30.11.2020
"""

import torch, os, sys, glob
cwd = os.getcwd()
sys.path.append(cwd)
from tqdm import tqdm
import numpy as np
from lib.utils import load_obj, natural_key,setup_seed 
from lib.benchmark_utils import ransac_pose_estimation, get_inlier_ratio, get_scene_split, write_est_trajectory
import open3d as o3d
from lib.benchmark import read_trajectory, write_trajectory, benchmark
import argparse
setup_seed(0)

def sample_interest_points(method, scores, N):
    """
    We can do random sampling, probabilistic sampling, or top-k sampling
    """
    assert method in ['prob','topk', 'random']
    n = scores.size(0)
    if n < N:
        choice = np.random.choice(n, N)
    else:
        if method == 'random':
            choice = np.random.permutation(n)[:N]
        elif method =='topk':
            choice = torch.topk(scores, N, dim=0)[1]
        elif method =='prob':
            idx = np.arange(n)
            probs = (scores / scores.sum()).numpy().flatten()
            choice = np.random.choice(idx, size= N, replace=False, p=probs)
    
    return choice



def benchmark_predator(feats_scores,n_points,exp_dir,whichbenchmark,sample_method,ransac_with_mutual=False, inlier_ratio_threshold = 0.05):
    gt_folder = f'configs/benchmarks/{whichbenchmark}'
    exp_dir = f'{exp_dir}/{whichbenchmark}_{n_points}_{sample_method}'
    if(not os.path.exists(exp_dir)):
        os.makedirs(exp_dir)
    print(exp_dir)

    results = dict()
    results['w_mutual'] = {'inlier_ratios':[], 'distances':[]}
    results['wo_mutual'] = {'inlier_ratios':[], 'distances':[]}
    tsfm_est = []
    for eachfile in tqdm(feats_scores):
        ########################################
        # 1. take the input point clouds
        data = torch.load(eachfile)
        len_src =  data['len_src']
        pcd =  data['pcd']
        feats =  data['feats']
        rot, trans = data['rot'], data['trans']
        saliency, overlap =  data['saliency'], data['overlaps']

        src_pcd = pcd[:len_src]
        tgt_pcd = pcd[len_src:]
        src_feats = feats[:len_src]
        tgt_feats = feats[len_src:]
        src_overlap, src_saliency = overlap[:len_src], saliency[:len_src]
        tgt_overlap, tgt_saliency = overlap[len_src:], saliency[len_src:]

        ########################################
        # 2. do probabilistic sampling guided by the score
        src_scores = src_overlap * src_saliency
        tgt_scores = tgt_overlap * tgt_saliency


        src_idx = sample_interest_points(sample_method, src_scores,n_points)
        tgt_idx = sample_interest_points(sample_method, tgt_scores,n_points)

        src_pcd, src_feats = src_pcd[src_idx], src_feats[src_idx]
        tgt_pcd, tgt_feats = tgt_pcd[tgt_idx], tgt_feats[tgt_idx]
        
        ########################################
        # 3. run ransac
        tsfm_est.append(ransac_pose_estimation(src_pcd, tgt_pcd, src_feats, tgt_feats, mutual=ransac_with_mutual))

        ########################################
        # 4. calculate inlier ratios
        inlier_ratio_results = get_inlier_ratio(src_pcd, tgt_pcd, src_feats, tgt_feats, rot, trans)

        results['w_mutual']['inlier_ratios'].append(inlier_ratio_results['w']['inlier_ratio'])
        results['w_mutual']['distances'].append(inlier_ratio_results['w']['distance'])
        results['wo_mutual']['inlier_ratios'].append(inlier_ratio_results['wo']['inlier_ratio'])
        results['wo_mutual']['distances'].append(inlier_ratio_results['wo']['distance'])

    tsfm_est = np.array(tsfm_est)

    ########################################
    # wirte the estimated trajectories
    write_est_trajectory(gt_folder, exp_dir, tsfm_est)
    
    ########################################
    # evaluate the results, here FMR and Inlier ratios are all average twice
    benchmark(exp_dir, gt_folder)
    split = get_scene_split(whichbenchmark)

    for key in['w_mutual','wo_mutual']:
        inliers =[]
        fmrs = []

        for ele in split:
            c_inliers = results[key]['inlier_ratios'][ele[0]:ele[1]]
            inliers.append(np.mean(c_inliers))
            fmrs.append((np.array(c_inliers) > inlier_ratio_threshold).mean())

        with open(os.path.join(exp_dir,'result'),'a') as f:
            f.write(f'Inlier ratio {key}: {np.mean(inliers):.3f} : +- {np.std(inliers):.3f}\n')
            f.write(f'Feature match recall {key}: {np.mean(fmrs):.3f} : +- {np.std(fmrs):.3f}\n')
        f.close()


if __name__=='__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source_path', default=None, type=str, help='path to precomputed features and scores')
    parser.add_argument(
        '--benchmark', default='3DLoMatch', type=str, help='[3DMatch, 3DLoMatch]')
    parser.add_argument(
        '--n_points', default=1000, type=int, help='number of points used by RANSAC')
    parser.add_argument(
        '--exp_dir', default='est_traj', type=str, help='export final results')
    parser.add_argument(
        '--sampling', default='prob', type = str, help='interest point sampling')
    args = parser.parse_args()

    feats_scores = sorted(glob.glob(f'{args.source_path}/*.pth'), key=natural_key)

    benchmark_predator(feats_scores, args.n_points, args.exp_dir, args.benchmark, args.sampling)
