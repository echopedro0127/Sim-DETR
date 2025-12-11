#!/bin/bash
# Quick training script (5 epochs) for Sim-DETR with InternVideo2 features

dset_name=hl
ctx_mode=video_tef
v_feat_types=internvideo2  # Use InternVideo2 features
t_feat_type=internvideo2   # Use InternVideo2 text features
results_root=results_internvideo2_quick
exp_id=internvideo2_5epoch_$(date +%Y%m%d_%H%M%S)

######## data paths
train_path=data/highlight_train_release.jsonl
eval_path=data/highlight_val_release.jsonl
eval_split_name=val

######## setup video+text features
# Use SDST dataset features
feat_root=/home/cl/vscode/sdst_datasets/data/qvhighlights

# video features - InternVideo2
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"internvideo2"* ]]; then
  v_feat_dirs+=(${feat_root}/vid_feats)
  # InternVideo2 features: shape (L, K, D) = (75, 5, 768)
  # After reshape: (L, K*D) = (75, 3840)
  (( v_feat_dim += 3840 ))
fi

# text features - InternVideo2
if [[ ${t_feat_type} == "internvideo2" ]]; then
  t_feat_dir=${feat_root}/txt_feats
  # InternVideo2 text features: shape (K, T, D) = (5, 17, 1024)
  # After reshape: (T, K*D) = (17, 5120)
  t_feat_dim=5120
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

#### training - Quick 5 epoch setup
bsz=32
lr_drop=3              # LR decay at epoch 3 (for 5 epoch training)
lr=0.0001
n_epoch=5              # Only 5 epochs for quick training
max_es_cnt=5           # Early stopping after 5 epochs
lw_saliency=1.0
seed=2017
VTC_loss_coef=0.3
CTC_loss_coef=0.5
label_loss_coef=4
dec_layers=4
enc_layers=2

echo "=========================================="
echo "Starting Quick Training (5 epochs)"
echo "=========================================="
echo "Experiment ID: ${exp_id}"
echo "Results will be saved to: ${results_root}/${exp_id}"
echo "Video feature dim: ${v_feat_dim}"
echo "Text feature dim: ${t_feat_dim}"
echo "=========================================="

PYTHONPATH=$PYTHONPATH:. python sim_detr/train.py \
--seed $seed \
--label_loss_coef $label_loss_coef \
--VTC_loss_coef $VTC_loss_coef \
--CTC_loss_coef $CTC_loss_coef \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--lr ${lr} \
--n_epoch ${n_epoch} \
--max_es_cnt ${max_es_cnt} \
--lw_saliency ${lw_saliency} \
--lr_drop ${lr_drop} \
--dec_layers ${dec_layers} \
--enc_layers ${enc_layers} \
${@:1}
