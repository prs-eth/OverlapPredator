## PREDATOR: Registration of 3D Point Clouds with Low Overlap
This repository provides code and data required to train and evaluate PREDATOR, a  model  for  **p**airwise point-cloud **re**gistration with **d**eep **at**tention to the **o**verlap **r**egion. It represents the official implementation of the paper:

### [PREDATOR: Registration of 3D Point Clouds with Low Overlap](https://arxiv.org/abs/2011.13005)

\*[Shengyu Huang](https://shengyuh.github.io), \*[Zan Gojcic](https://zgojcic.github.io/), [Mikhail Usvyatsov](https://aelphy.github.io), [Andreas Wieser](https://gseg.igp.ethz.ch/people/group-head/prof-dr--andreas-wieser.html), [Konrad Schindler](https://prs.igp.ethz.ch/group/people/person-detail.schindler.html)\
|[ETH Zurich](https://igp.ethz.ch/) |\
\* Equal contribution

![Predator_teaser](figures/teaser_predator.jpg?raw=true)



### Contact
If you have any questions, please let us know: Shengyu Huang {shengyu.huang@geod.baug.ethz.ch}, Zan Gojcic {zan.gojcic@geod.baug.ethz.ch}

## News
- 2020-11-30: Code and paper release

## Instructions
This code has been tested on 
- Python 3.6.9/3.7.4, PyTorch 1.4.0/1.5.1, CUDA 10.1/11.0, gcc 6.3/7.5, TITAN Xp/GeForce RTX 2080 Ti/GeForce GTX 1080Ti
- Python 3.8.5, PyTorch 1.8.0.dev20201124+cu110, CUDA 11.1, gcc 9.3.0, GeForce RTX 3090

**Note**: We observed random data loader crashes due to memory issues on machines with less than 64GB CPU RAM. If you observe similar issues, please consider reducing the number of workers. 

### Requirements
To create a virtual environment and install the required dependences please run:
```shell
git clone https://github.com/ShengyuH/OverlapPredator.git
virtualenv --no-site-packages predator -p python3; source predator/bin/activate
cd OverlapPredator; pip install -r requirements.txt
cd cpp_wrappers; sh compile_wrappers.sh; cd ..
```
in your working folder.

### Datasets and pretrained models
We provide preprocessed 3DMatch pairwise datasets (voxel-grid subsampled fragments together with their ground truth transformation matrices), and two pretrained models on 3DMatch dataset. The preprocessed data and models can be downloaded by running:
```shell
sh scripts/download_data_weight.sh
```

Predator is the model evaluated in the paper whereas bigPredator is a wider network that is trained on a single GeForce RTX 3090. 

| Model       | first_feats_dim   | gnn_feats_dim | # parameters|
|:-----------:|:-------------------:|:-------:|:-------:|
| Predator | 128               | 256 | 7.43M|
| bigPredator | 256                | 512 | 29.67M|

The results of both Predator and bigPredator, obtained using the evaluation protocol described in the paper, are available in the bottom table:

<img src="figures/results.png" alt="results" width="500"/>

**Note**: The pretrained models and processed data of ModelNet will be released in the following weeks. 

### Train
After creating the virtual environment and downloading the datasets, Predator can be trained using:
```shell
python main.py --mode train --exp_dir predator_3dmatch --first_feats_dim 128 --gnn_feats_dim 256
```
and biGPREDATOR using: 
```shell
python main.py --mode train --exp_dir bigpredator_3dmatch --first_feats_dim 256 --gnn_feats_dim 512
```

### Evaluate
To evaluate PREDATOR, the first step is to extract features and overlap/matachability scores by running: 
```shell
python main.py --mode test --exp_dir predator_3dmatch --pretrain weights/Predator.pth --first_feats_dim 128 --gnn_feats_dim 256 --test_info 3DLoMatch
```
the features will be saved to ```snapshot/{exp_dir}/{test_info}```. The estimation of the transformation parameters using RANSAC can then be carried out using:
```shell
python scripts/evaluate_predator.py --source_path snapshot/predator_3dmatch/3DLoMatch --n_points 1000 --benchmark 3DLoMatch --exp_dir est_3dlomatch_1000
```
dependent on ```n_points``` used by RANSAC, this might take a few minutes. The final results are stored in ```est_{test_info}_{n_points}/result```. To evaluate PREDATOR on 3DMatch benchmark, simply replace 3DLoMatch by 3DMatch.

### 
### Demo
We prepared a small demo, which demonstrates the whole Predator pipeline using two random fragments from the 3DMatch dataset. To carry out the demo, please run:
```shell
python scripts/demo.py
```

The demo script will visualize input point clouds, inferred overlap regions, and point cloud aligned with the estimated transformation parameters.

### Citation

If you find this code useful for your work or use it in your project, please consider citing:

```shell
@article{huang2020predator,
  title={PREDATOR: Registration of 3D Point Clouds with Low Overlap},
  author={Shengyu Huang, Zan Gojcic, Mikhail Usvyatsov, Andreas Wieser, Konrad Schindler},
  journal={arXiv:2011.13005 [cs.CV]},
  year={2020}
}
```

### Acknowledgments
In this project we use (parts of) the official implementations of the followin works: 

- [FCGF](https://github.com/chrischoy/FCGF)
- [D3Feat](https://github.com/XuyangBai/D3Feat.pytorch)
- [3DSmoothNet](https://github.com/zgojcic/3DSmoothNet)
- [MultiviewReg](https://github.com/zgojcic/3D_multiview_reg)
- [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)
- [DGCNN](https://github.com/WangYueFt/dgcnn)
- [RPMNet](https://github.com/yewzijian/RPMNet)
- [KPConv](https://github.com/HuguesTHOMAS/KPConv-PyTorch)

 We thank the respective authors for open sourcing their methods.