# note: only include the results in kitti dataset, because the pre-compute results on kaist is too large
# all of the results are stored in the folder: "train_log/attention_12_16_hidden_cell_mix_fbatt"

# collect feature active
python feature_analyse/static_feature.py --gpu_id 0

# run shap (very slow if run on all, so the example is only on kitti using ORB-SLAM_mono)
python feature_analyse/shap_explainer.py --gpu_id 0

# general the grid map (Fig. 2(a)) and effect of feature (Fig. 2(b))
python feature_analyse/grid_score.py --plot --shap

# general the head-direction map (Fig. 2(a)) and effect of feature (Fig. 2(b))
python feature_analyse/hd_score.py --plot --shap