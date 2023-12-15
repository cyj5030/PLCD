## Download code
```
git clone git@github.com:cyj5030/PLCD.git
```

## Requirements
The code has been tested with PyTorch 1.11 and Cuda 11.3.

```
conda create --name plcd python=3.10
conda activate plcd
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install pandas=1.5.2 matplotlib=3.5.1 scipy=1.7.3 scikit-image=0.19.3 numba=0.56.4 astropy=5.1 pyyaml=6.0 seaborn=0.12.1
pip install opencv-python==4.5.5.64 shap==0.41.0 einops==0.4.1
```

## Data Download 
0. Download the pre-compute results of [BoTW in KITTI](https://drive.google.com/file/d/1Sx4pHC9ETBBxHUj7FH-L2a5l_IObuAy9/view?usp=sharing), [BoW in KITTI](https://drive.google.com/file/d/1x5HT2bFEzfVyUsTFpIOqiA1-u9YWcykY/view?usp=sharing), [visual label of KITTI](https://drive.google.com/file/d/1OsbbNauSgsw_lxc8L_fHypNzTR5tCSjD/view?usp=sharing) and [trajectories](https://drive.google.com/file/d/1YPftCLExDdKzfKGgUFoFbUivyLRn9PGN/view?usp=sharing).

1. unzip the package to the folder of "data"
```
unzip BoTW_in_KITTI.zip -d data/
unzip BoW_in_KITTI.zip -d data/
unzip trajectory.zip -d data/
unzip visual_label_of_kitti.zip -d data/
```

2. Download the [pre-trained model](https://drive.google.com/file/d/1C6FNQqDQ8p25Du2hvAvLFVJckf42D6of/view?usp=sharing).

3. unzip the package to the root
```
unzip checkpoint.zip
```

## Inference
Note: only include the results in kitti dataset, because the pre-compute results on kaist is too large. All of the visualization results are stored in the folder: "train_log/attention_12_16_hidden_cell_mix_fbatt"

### Quantitative results
For compute all of the quantitative results, run the script in:
```
./evaluation_scripts/eval.sh
```

If you want to run separate, run the following commands:
* To compute the results of Table 1 in folder: "train_log/attention_12_16_hidden_cell_mix_fbatt/VO_evals", run
```
python evaluation_scripts/eval_vo.py --gpu_id 0
```

* To compute the results of Table 2 in folder: "train_log/attention_12_16_hidden_cell_mix_fbatt/VLCD_evals", run
```
python evaluation_scripts/eval_visual.py --gpu_id 0
```

* To compute the results of Table A1 in folder: "train_log/attention_12_16_hidden_cell_mix_fbatt/Merge_evals_BoW", run
```
python evaluation_scripts/eval_merge_bow.py --gpu_id 0
```

* To compute the results of Table A1 in folder: "train_log/attention_12_16_hidden_cell_mix_fbatt/Merge_evals_BoTW", run
```
python evaluation_scripts/eval_merge_botw.py --gpu_id 0
```

### Visualization
For compute all of the quantitative results, run the script in:
```
./evaluation_scripts/visualization.sh
```

If you want to run separate, run the following commands:
* To compute the visualization of pose ground-truth, run
```
python evaluation_scripts/vis_traj_gt.py --gpu_id 0
```

* To compute the visualization of plcd results, run
```
python evaluation_scripts/vis_traj_plcd.py --gpu_id 0
```

* To compute the visualization of bow results, run
```
python evaluation_scripts/vis_traj_bow.py --gpu_id 0
```

* To compute the visualization of botw results, run
```
python evaluation_scripts/vis_traj_botw.py --gpu_id 0
```

* To compute the visualization of bow+PLCD-GVO, run
```
python evaluation_scripts/vis_traj_merge_bow.py --gpu_id 0
```

* To compute the visualization of botw+PLCD-GVO, run
```
python evaluation_scripts/vis_traj_merge_botw.py --gpu_id 0
```

### Feature analyse
For compute all of the results on features, run the script in:
```
./feature_analyse/feature.sh
```

If you want to run separate, run the following commands step by step:
* First, collect feature active
```
python feature_analyse/static_feature.py --gpu_id 0
```

* Then, compute shap value (very slow if run on all datasets, so the example is only on kitti using ORB-SLAM_mono)
```
python feature_analyse/shap_explainer.py --gpu_id 0
```

* General the grid map (Fig. 2(a)) and effect of feature (Fig. 2(b))
```
python feature_analyse/grid_score.py --plot --shap
```

* General the head-direction map (Fig. 2(a)) and effect of feature (Fig. 2(b))
```
python feature_analyse/hd_score.py --plot --shap
```

## Related Projects
[Grid_cell_analysis](https://github.com/MattNolanLab/grid_cell_analysis) 

