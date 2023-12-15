# note: only include the results in kitti dataset, because the pre-compute results on kaist is too large
# all of the visualization results are stored in the folder: "train_log/attention_12_16_hidden_cell_mix_fbatt/plots"

# visualization of pose ground-truth
python evaluation_scripts/vis_traj_gt.py --gpu_id 0

# visualization of plcd results
python evaluation_scripts/vis_traj_plcd.py --gpu_id 0

# visualization of bow results
python evaluation_scripts/vis_traj_bow.py --gpu_id 0

# visualization of botw results
python evaluation_scripts/vis_traj_botw.py --gpu_id 0

# visualization of bow+PLCD-GVO
python evaluation_scripts/vis_traj_merge_bow.py --gpu_id 0

# visualization of botw+PLCD-GVO
python evaluation_scripts/vis_traj_merge_botw.py --gpu_id 0