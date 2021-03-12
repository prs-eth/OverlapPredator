## PREDATOR: Registration of 3D Point Clouds with Low Overlap (CVPR 2021, Oral)
This repository represents the official implementation of the paper:

### [PREDATOR: Registration of 3D Point Clouds with Low Overlap](https://arxiv.org/abs/2011.13005)

\*[Shengyu Huang](https://shengyuh.github.io), \*[Zan Gojcic](https://zgojcic.github.io/), [Mikhail Usvyatsov](https://aelphy.github.io), [Andreas Wieser](https://gseg.igp.ethz.ch/people/group-head/prof-dr--andreas-wieser.html), [Konrad Schindler](https://prs.igp.ethz.ch/group/people/person-detail.schindler.html)\
|[ETH Zurich](https://igp.ethz.ch/) | \* Equal contribution

For implementation using MinkowskiEngine backbone, please check [this](https://github.com/ShengyuH/OverlapPredator.Mink)

For more information, please see the [project website](https://overlappredator.github.io)

![Predator_teaser](assets/teaser_predator.jpg?raw=true)



### Contact
If you have any questions, please let us know: 
- Shengyu Huang {shengyu.huang@geod.baug.ethz.ch}
- Zan Gojcic {zan.gojcic@geod.baug.ethz.ch}

## News
- 2021-02-28: MinkowskiEngine-based PREDATOR [release](https://github.com/ShengyuH/OverlapPredator.Mink.git)
- 2021-02-23: Modelnet and KITTI release
- 2020-11-30: Code and paper release

## Instructions
This code has been tested on 
- Python 3.8.5, PyTorch 1.7.1, CUDA 11.2, gcc 9.3.0, GeForce RTX 3090/GeForce GTX 1080Ti

**Note**: We observe random data loader crashes due to memory issues, if you observe similar issues, please consider reducing the number of workers or increasing CPU RAM. We now released a sparse convolution-based Predator, have a look [here](https://github.com/ShengyuH/OverlapPredator.Mink.git)!

### Requirements
To create a virtual environment and install the required dependences please run:
```shell
git clone https://github.com/ShengyuH/OverlapPredator.git
virtualenv predator; source predator/bin/activate
cd OverlapPredator; pip install -r requirements.txt
cd cpp_wrappers; sh compile_wrappers.sh; cd ..
```
in your working folder.

### Datasets and pretrained models
For KITTI dataset, please follow the instruction on [KITTI Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) to download the KITTI odometry training set.

We provide 
- preprocessed 3DMatch pairwise datasets (voxel-grid subsampled fragments together with their ground truth transformation matrices)
- modelnet dataset
- pretrained models on 3DMatch, KITTI and Modelnet

The preprocessed data and models can be downloaded by running:
```shell
sh scripts/download_data_weight.sh
```

Predator is the model evaluated in the paper whereas bigPredator is a wider network that is trained on a single GeForce RTX 3090. 

| Model       | first_feats_dim   | gnn_feats_dim | # parameters|
|:-----------:|:-------------------:|:-------:|:-------:|
| Predator | 128               | 256 | 7.43M|
| bigPredator | 256                | 512 | 29.67M|

The results of both Predator and bigPredator, obtained using the evaluation protocol described in the paper, are available in the bottom table:

<img src="assets/results.png" alt="results" width="450"/>

### 3DMatch(Indoor)
#### Train
After creating the virtual environment and downloading the datasets, Predator can be trained using:
```shell
python main.py configs/train/indoor.yaml
```

#### Evaluate
For 3DMatch, to reproduce Table 2 in our main paper, we first extract features and overlap/matachability scores by running: 
```shell
python main.py configs/test/indoor.yaml
```
the features will be saved to ```snapshot/indoor/3DMatch```. The estimation of the transformation parameters using RANSAC can then be carried out using:
```shell
for N_POINTS in 250 500 1000 2500 5000
do
  python scripts/evaluate_predator.py --source_path snapshot/indoor/3DMatch --n_points $N_POINTS --benchmark 3DMatch
done
```
dependent on ```n_points``` used by RANSAC, this might take a few minutes. The final results are stored in ```est_traj/{benchmark}/{n_points}/result```. To evaluate PREDATOR on 3DLoMatch benchmark, please also change ```3DMatch``` to ```3DLoMatch``` in ```configs/test/indoor.yaml```.

#### Demo
We prepared a small demo, which demonstrates the whole Predator pipeline using two random fragments from the 3DMatch dataset. To carry out the demo, please run:
```shell
python scripts/demo.py configs/test/indoor.yaml
```

The demo script will visualize input point clouds, inferred overlap regions, and point cloud aligned with the estimated transformation parameters:

<img src="assets/demo.png" alt="demo" width="750"/>

### KITTI(Outdoor)
We provide a small script to evaluate Predator on KITTI test set, after configuring KITTI dataset, please run:
```
python main.py configs/test/kitti.yaml
```
the results will be saved to the log file.

### ModelNet(Synthetic)
We provide a small script to evaluate Predator on ModelNet test set, please run:
```
python main.py configs/test/modelnet.yaml
```
The rotation and translation errors could be better/worse than the reported ones due to randomness in RANSAC. 

### Citation
If you find this code useful for your work or use it in your project, please consider citing:

```shell
@article{huang2020predator,
  title={PREDATOR: Registration of 3D Point Clouds with Low Overlap},
  author={Shengyu Huang, Zan Gojcic, Mikhail Usvyatsov, Andreas Wieser, Konrad Schindler},
  journal={CVPR},
  year={2021}
}
```

### Acknowledgments
In this project we use (parts of) the official implementations of the followin works: 

- [FCGF](https://github.com/chrischoy/FCGF) (KITTI preprocessing)
- [D3Feat](https://github.com/XuyangBai/D3Feat.pytorch) (KPConv backbone)
- [3DSmoothNet](https://github.com/zgojcic/3DSmoothNet) (3DMatch preparation)
- [MultiviewReg](https://github.com/zgojcic/3D_multiview_reg) (3DMatch benchmark)
- [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) (Transformer part)
- [DGCNN](https://github.com/WangYueFt/dgcnn) (self-gnn)
- [RPMNet](https://github.com/yewzijian/RPMNet) (ModelNet preprocessing and evaluation)

 We thank the respective authors for open sourcing their methods. We would also like to thank reviewers, especially reviewer 2 for his/her valuable inputs. 
