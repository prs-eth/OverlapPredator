# PREDATOR: Registration of 3D Point Clouds with Low Overlap
This repository provides code and data required to train and evaluate PREDATOR, a  model  for  **p**airwise point-cloud **re**gistration with **d**eep **at**tention to the **o**verlap **r**egion. It represents the official implementation of the paper:

### [PREDATOR: Registration of 3D Point Clouds with Low Overlap](https://addlink)

\*[Shengyu Huang](https://shengyuh.github.io)
, \*[Zan Gojcic](https://zgojcic.github.io/)
, [Mikhail Usvyatsov](https://aelphy.github.io)
, [Andreas Wieser](https://gseg.igp.ethz.ch/people/group-head/prof-dr--andreas-wieser.html)
, [Konrad Schindler](https://prs.igp.ethz.ch/group/people/person-detail.schindler.html)\
|[ETH Zurich](https://igp.ethz.ch/) |\
\* Equal contribution



![Predator_teaser](figures/teaser_predator.jpg?raw=true)


### Citation

If you find this code useful for your work or use it in your project, please consider citing:

```shell
add bibtex
```

### Contact
If you have any questions, please let us know: Shengyu Huang {shengyu.huang@geod.baug.ethz.ch}

## News
- 2020-11-30: Code release

## Instructions
This code has been tested on 
- Python 3.6.9, PyTorch 1.4.0, CUDA 11.0, gcc 7.5, TITAN Xp
- Python 3.7.4, PyTorch 1.5.1+cu101, CUDA 10.1, gcc 6.3.0, GeForce RTX 2080 Ti
- Python 3.8.5, PyTorch 1.8.0.dev20201124+cu110, CUDA 11.1, gcc 9.3.0, GeForce RTX 3090

**Note**: We observe data loader random crash due to memory issues on machines with less than 64GB CPU RAM.

### Requirements
Under your working folder, our virtual environment and requirements can be installed by running:
```shell
git clone https://github.com/ShengyuH/OverlapPredator.git
virtualenv --no-site-packages predator -p python3; source predator/bin/activate
cd OverlapPredator; pip install -r requirements.txt
cd cpp_wrappers; sh compile_wrappers.sh; cd ..
```

### Datasets
We provide preprocessed 3DMatch pairwise datasets, you can download them [here](https://drive.google.com/file/d/11oD5YsLn4OBNpLp4d-VEZtHegWpHaa_K/view?usp=sharing)(500MB). Please unzip it and move to ```OverlapPredator```.

### Pretrained weights
We provide two pretrained models on 3DMatch dataset, Predator and bigPredator. bigPredator is a wider network which is trained on a single GeForce RTX 3090, you can download them [here](https://drive.google.com/file/d/1xLqv1CBiFukRUn7fHYiLTXGuc0q1xo3x/view?usp=sharing)(275MB). Please unzip it and move to ```OverlapPredator```.

| Model       | first_feats_dim   | gnn_feats_dim | # parameters| performance |
|:-----------:|:-------------------:|:-------:|:-------:|:-------:|
| Predator | 128               | 256 | 7.43M| recall: 34%|
| bigPredator | 256                | 512 | 29.67M| recall: 40%|


### Train
After having virtual environment and datasets prepared, you can train from scratch by running:
```shell
python main.py --mode train --exp_dir predator_3dmatch --first_feats_dim 128 --gnn_feats_dim 256
```
or 
```shell
python main.py --mode train --exp_dir bigpredator_3dmatch --first_feats_dim 256 --gnn_feats_dim 512
```

### Evaluate
To evaluate PREDATOR, we first extract features and scores and store them as .pth files, then run RANSAC. To extract features and scores on 3DLoMatch benchmark, run: 
```shell
python main.py --mode test --exp_dir val_predator --pretrain model_zoo/Predator.pth --first_feats_dim 128 --gnn_feats_dim 256 --test_info 3DLoMatch
```
the features will be saved to ```snapshot/{exp_dir}/3DLoMatch```. Then we can run RANSAC by:
```shell
python evaluate_predator.py --source_path snapshot/val_predator/3DLoMatch --n_points 1000 --exp_dir est_3dlomatch_1000
```
this might take a few minutes, depends on ```n_points``` used by RANSAC. 

### 
### Demo
TODO





