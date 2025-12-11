# Sim-DETR with InternVideo2 Features

This document describes how to train Sim-DETR using InternVideo2 features extracted from the SDST dataset.

## Overview

This setup allows you to train Sim-DETR with InternVideo2 features instead of the original CLIP+SlowFast features. InternVideo2 is a more powerful video understanding model that may improve performance on temporal sentence grounding tasks.

## Feature Comparison

| Feature Type | Original (CLIP+SlowFast) | InternVideo2 |
|--------------|-------------------------|--------------|
| **Video Features** | CLIP: (L, 4, 768) → 3072D<br>SlowFast: 2304D<br>**Total: 5376D** | InternVideo2: (L, 5, 768) → **3840D** |
| **Text Features** | CLIP: (T, 4, 512) → **2048D** | InternVideo2: (5, T, 1024) → **5120D** |
| **Total Input Dim** | **7424D** (5376 + 2048) | **8960D** (3840 + 5120) |

### InternVideo2 Feature Format

- **Video features**: `.npz` files with key `'visual_feats_interm_pooled'`
  - Shape: `(L, K, D)` = `(75, 5, 768)`
  - L: video length (number of frames)
  - K: number of intermediate layers (5)
  - D: feature dimension (768)
  - After reshape: `(L, K*D)` = `(75, 3840)`

- **Text features**: `.npz` files with key `'text_feats'`
  - Shape: `(K, T, D)` = `(5, 17, 1024)`
  - K: number of intermediate layers (5)
  - T: text token length (max 17)
  - D: feature dimension (1024)
  - After reshape: `(T, K*D)` = `(17, 5120)`

## Setup Instructions

### 1. Environment Setup

The virtual environment has already been created and configured:

```bash
cd /home/cl/vscode/rmodel/Sim-DETR
source simdetr_env/bin/activate
```

### 2. Verify Installation

Test that InternVideo2 features can be loaded correctly:

```bash
python test_internvideo2_loading.py
```

Expected output:
```
✓ All tests passed successfully!
✓ Text feature dimension correct! (5120)
✓ Video feature dimension correct! (3842)  # 3840 + 2 for TEF
```

### 3. Data Paths

The training script is configured to use features from:
- **Video features**: `/home/cl/vscode/sdst_datasets/data/qvhighlights/vid_feats/`
- **Text features**: `/home/cl/vscode/sdst_datasets/data/qvhighlights/txt_feats/`
- **Annotations**: `data/highlight_train_release.jsonl` and `data/highlight_val_release.jsonl`

## Training

### Quick Start

To train Sim-DETR with InternVideo2 features:

```bash
cd /home/cl/vscode/rmodel/Sim-DETR
source simdetr_env/bin/activate
bash sim_detr/scripts/train_internvideo2.sh
```

### Training Configuration

The training script ([sim_detr/scripts/train_internvideo2.sh](sim_detr/scripts/train_internvideo2.sh)) uses the following configuration:

```bash
# Feature dimensions
v_feat_dim=3840      # InternVideo2 video: 5 layers × 768D
t_feat_dim=5120      # InternVideo2 text: 5 layers × 1024D

# Training hyperparameters
bsz=32               # Batch size
lr=0.0001            # Learning rate
n_epoch=200          # Number of epochs
lr_drop=100          # LR decay at epoch 100
lw_saliency=1.0      # Saliency loss weight
VTC_loss_coef=0.3    # Video-Text Contrastive loss
CTC_loss_coef=0.5    # Cross-modal Temporal Contrastive loss

# Model architecture
dec_layers=4         # Decoder layers
enc_layers=2         # Encoder layers
```

### Custom Training

To modify training parameters, you can either:

1. **Edit the training script**:
   ```bash
   vim sim_detr/scripts/train_internvideo2.sh
   ```

2. **Pass arguments directly**:
   ```bash
   bash sim_detr/scripts/train_internvideo2.sh --bsz 16 --lr 0.0002
   ```

## Model Architecture Modifications

The key modification is in the dataset loader ([sim_detr/start_end_dataset_internvideo2.py](sim_detr/start_end_dataset_internvideo2.py)):

### Video Feature Loading

```python
def _get_video_feat_by_vid(self, vid):
    # Load InternVideo2 features
    _feat_path = join(_feat_dir, f"{vid}.npz")
    _feat_data = np.load(_feat_path)
    _feat = _feat_data['visual_feats_interm_pooled'][:self.max_v_l]  # (L, K, D)

    # Reshape from (L, K, D) to (L, K*D)
    l = len(_feat)
    _feat = _feat.reshape(l, -1)  # (L, 3840)

    return torch.from_numpy(_feat)
```

### Text Feature Loading

```python
def _get_query_feat_by_qid(self, qid):
    # Load InternVideo2 text features
    q_feat_path = join(self.q_feat_dir, f"{qid}.npz")
    q_feat_data = np.load(q_feat_path)
    q_feat = q_feat_data['text_feats']  # (K, T, D)

    # Reshape from (K, T, D) to (T, K*D)
    K, T, D = q_feat.shape
    q_feat = q_feat.transpose(1, 0, 2)  # (T, K, D)
    q_feat = q_feat[:self.max_q_l]
    l = len(q_feat)
    q_feat = q_feat.reshape(l, -1)  # (T, 5120)

    return torch.from_numpy(q_feat)
```

## Expected Performance

### Baseline (CLIP + SlowFast)

According to the Sim-DETR paper, the baseline performance on QVHighlights is:
- **Moment Retrieval**: R1@0.5 ≈ 62-65%, mAP ≈ 40-43%
- **Highlight Detection**: mAP ≈ 38-40%

### InternVideo2 Hypothesis

InternVideo2 features are expected to provide:
1. **Better temporal modeling**: 5 intermediate layers capture multi-scale temporal information
2. **Richer semantic features**: Larger feature dimension (3840D vs 3072D for video)
3. **Improved text-video alignment**: Unified feature extraction from the same model

**Expected improvements**: +2-5% on R1@0.5 and mAP metrics.

## Evaluation

After training, evaluate the model:

```bash
# Generate validation predictions
bash sim_detr/scripts/inference.sh results_internvideo2/{exp_dir}/model_best.ckpt 'val'

# Generate test predictions
bash sim_detr/scripts/inference.sh results_internvideo2/{exp_dir}/model_best.ckpt 'test'
```

This will generate:
- `hl_val_submission.jsonl` - Validation set predictions
- `hl_test_submission.jsonl` - Test set predictions (for Codalab submission)

## Troubleshooting

### Issue: NumPy version error

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

**Solution**: Downgrade NumPy
```bash
pip install "numpy<2"
```

### Issue: Missing dependencies

```
ModuleNotFoundError: No module named 'pandas'
```

**Solution**: Install missing packages
```bash
pip install pandas tqdm tensorboard h5py easydict scipy
```

### Issue: Feature dimension mismatch

If you see dimension mismatch errors, verify:
1. Feature files are in the correct format (`.npz` with correct keys)
2. Feature dimensions match expected values:
   - Video: `(L, 5, 768)` → reshape to `(L, 3840)`
   - Text: `(5, T, 1024)` → reshape to `(T, 5120)`

## Files Created

This setup includes the following new files:

1. **[sim_detr/start_end_dataset_internvideo2.py](sim_detr/start_end_dataset_internvideo2.py)** - Dataset loader with InternVideo2 support
2. **[sim_detr/scripts/train_internvideo2.sh](sim_detr/scripts/train_internvideo2.sh)** - Training script for InternVideo2
3. **[test_internvideo2_loading.py](test_internvideo2_loading.py)** - Feature loading test script
4. **[README_INTERNVIDEO2.md](README_INTERNVIDEO2.md)** - This documentation

## Comparison with Original Sim-DETR

| Aspect | Original Sim-DETR | InternVideo2 Version |
|--------|------------------|---------------------|
| Video Encoder | CLIP ViT-B/32 | InternVideo2-1B |
| Text Encoder | CLIP ViT-B/32 | InternVideo2-1B |
| Video Dim | 3072D (CLIP) + 2304D (SlowFast) | 3840D (InternVideo2) |
| Text Dim | 2048D | 5120D |
| Feature Extraction | Separate models | Unified model |
| Training Time | Baseline | Similar (same architecture) |

## Next Steps

1. **Run full training**: Train for 200 epochs on QVHighlights
2. **Compare results**: Compare with baseline CLIP+SlowFast performance
3. **Ablation studies**:
   - Try different layer combinations (e.g., use only last 3 layers)
   - Experiment with feature pooling strategies
   - Test on other datasets (Charades-STA, TACoS)

## References

- **Sim-DETR Paper**: [arXiv:2509.23867](https://arxiv.org/abs/2509.23867)
- **InternVideo2**: State-of-the-art video understanding model
- **SDST**: Sparse-Dense Side-Tuner for temporal grounding

## Contact

For questions or issues, please refer to:
- Original Sim-DETR: [https://github.com/SooLab/Sim-DETR](https://github.com/SooLab/Sim-DETR)
- SDST project: `/home/cl/vscode/SDSTnew/SDST/`
