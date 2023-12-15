# note: only include the results in kitti dataset, because the pre-compute results on kaist is too large

# results of Table 1 in folder: "train_log/attention_12_16_hidden_cell_mix_fbatt/VO_evals"
python evaluation_scripts/eval_vo.py --gpu_id 0

# results of Table 2 in folder: "train_log/attention_12_16_hidden_cell_mix_fbatt/VLCD_evals"
python evaluation_scripts/eval_visual.py --gpu_id 0

# results of Table A1 in folder: "train_log/attention_12_16_hidden_cell_mix_fbatt/Merge_evals_BoW"
python evaluation_scripts/eval_merge_bow.py --gpu_id 0

# results of Table A1 in folder: "train_log/attention_12_16_hidden_cell_mix_fbatt/Merge_evals_BoTW"
python evaluation_scripts/eval_merge_botw.py --gpu_id 0
