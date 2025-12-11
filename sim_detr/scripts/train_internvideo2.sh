#!/bin/bash
# Training script for Sim-DETR with InternVideo2 features from SDST dataset

dset_name=hl
ctx_mode=video_tef
v_feat_types=internvideo2  # Use InternVideo2 features
t_feat_type=internvideo2   # Use InternVideo2 text features
results_root=results_internvideo2
exp_id=internvideo2_exp

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

#### training
bsz=32
lr_drop=100
lr=0.0001
n_epoch=200
lw_saliency=1.0
seed=2017
VTC_loss_coef=0.3
CTC_loss_coef=0.5
label_loss_coef=4
dec_layers=4
enc_layers=2

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
--lw_saliency ${lw_saliency} \
--lr_drop ${lr_drop} \
--dec_layers ${dec_layers} \
--enc_layers ${enc_layers} \
${@:1}
