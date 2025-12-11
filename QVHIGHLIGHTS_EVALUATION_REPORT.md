# Sim-DETR QVHighlights Evaluation Report

## Overview

This report summarizes the performance of the best Sim-DETR checkpoint on the QVHighlights validation set. The model was trained with InternVideo2 features and fine-tuned for 10 epochs from the best checkpoint at epoch 47.

**Best Checkpoint**: `model_best.ckpt` (Epoch 9 of fine-tuning round)
**Checkpoint Path**: `/home/cl/vscode/rmodel/Sim-DETR/results_internvideo2_finetune/hl-video_tef-finetune_from_epoch47_10ep-2025_12_11_20_54_10/model_best.ckpt`

---

## Training Progress (Fine-tuning Round 1: 10 Epochs)

| Epoch | R1@0.5 | R1@0.7 | mAP   | mAP@0.5 | mAP@0.75 | HL-Fair | HL-Good | HL-VGood |
|-------|--------|--------|-------|---------|----------|---------|---------|----------|
| 0     | 73.29% | 58.65% | 52.80%| 72.65%  | 54.65%   | 81.35%  | 70.15%  | 43.37%   |
| 1     | 73.23% | 59.29% | 52.28%| 72.28%  | 54.50%   | 81.65%  | 70.40%  | 43.56%   |
| 2     | 73.55% | 59.74% | 52.56%| 72.21%  | 54.95%   | 81.61%  | 70.45%  | 43.64%   |
| 3     | 73.10% | 60.19% | 52.75%| 72.31%  | 55.38%   | 81.75%  | 70.68%  | 43.87%   |
| 4     | 73.42% | 59.03% | 52.62%| 72.85%  | 54.93%   | 81.57%  | 70.53%  | 43.63%   |
| 5     | 73.42% | 60.06% | 53.11%| 72.22%  | 55.73%   | 81.51%  | 70.54%  | 43.71%   |
| 6     | 73.42% | 59.61% | 53.30%| 72.77%  | 55.47%   | 81.62%  | 70.55%  | 43.65%   |
| 7     | 73.87% | 59.74% | 52.81%| 72.82%  | 55.60%   | 81.69%  | 70.71%  | 43.87%   |
| 8     | 74.19% | 60.77% | 53.77%| 73.19%  | 56.25%   | 81.84%  | 70.65%  | 43.61%   |
| **9** | **74.71%** | **60.77%** | **53.87%** | **73.25%** | **56.66%** | **81.96%** | **70.76%** | **43.80%** |

**Best Performance**: Epoch 9 achieved the highest R1@0.5 score of 74.71%

---

## Best Model Performance (Epoch 9)

### Moment Retrieval (MR) Metrics

#### Overall Performance
- **R1@0.5**: 74.71%
- **R1@0.7**: 60.77%
- **mAP**: 53.87%
- **mAP@0.5**: 73.25%
- **mAP@0.75**: 56.66%

#### Recall@1 at Different IoU Thresholds
| IoU Threshold | R1 Score |
|---------------|----------|
| 0.5           | 74.71%   |
| 0.55          | 70.39%   |
| 0.6           | 66.77%   |
| 0.65          | 63.48%   |
| 0.7           | 60.77%   |
| 0.75          | 57.48%   |
| 0.8           | 51.87%   |
| 0.85          | 44.65%   |
| 0.9           | 34.90%   |
| 0.95          | 23.48%   |

#### mAP at Different IoU Thresholds
| IoU Threshold | mAP Score |
|---------------|-----------|
| 0.5           | 73.25%    |
| 0.55          | 68.98%    |
| 0.6           | 66.44%    |
| 0.65          | 63.14%    |
| 0.7           | 60.07%    |
| 0.75          | 56.66%    |
| 0.8           | 50.47%    |
| 0.85          | 43.08%    |
| 0.9           | 33.84%    |
| 0.95          | 22.80%    |
| **Average**   | **53.87%**|

### Performance by Video Length

#### Long Videos
- **mAP**: 58.29%
- **R1@0.5**: 70.03%
- **R1@0.7**: 58.01%

#### Middle-length Videos
- **mAP**: 57.83%
- **R1@0.5**: 72.31%
- **R1@0.7**: 59.35%

#### Short Videos
- **mAP**: 14.79%
- **R1@0.5**: 14.92%
- **R1@0.7**: 9.56%

**Note**: The model performs significantly better on long and middle-length videos compared to short videos.

---

### Highlight Detection (HL) Metrics

#### Overall Performance
- **Fair-mAP**: 81.96%
- **Fair-Hit@1**: 84.52%
- **Good-mAP**: 70.76%
- **Good-Hit@1**: 82.97%
- **VeryGood-mAP**: 43.80%
- **VeryGood-Hit@1**: 71.87%

**Interpretation**:
- The model achieves excellent performance on Fair-quality highlights (81.96% mAP)
- Good performance on Good-quality highlights (70.76% mAP)
- Moderate performance on VeryGood-quality highlights (43.80% mAP)

---

## Comparison with Training Logs Analysis

From the comprehensive training logs analysis (81 epochs total across all training phases):

### Training Timeline
1. **Initial Training (Epochs 1-21)**: Rapid improvement from 7.03% to 70.52% R1@0.5
2. **Continued Training (Epochs 22-50)**: Steady improvement, plateaued around 74.39% R1@0.5
3. **Fine-tuning Round 1 (Epochs 1-10)**: Achieved **BEST OVERALL** performance
   - **Epoch 10**: R1@0.5 = 74.71%, R1@0.7 = 60.77%, mAP = 53.87%
4. **Fine-tuning Round 2 (Epochs 1-10)**: Slight performance variations around 74.5%
5. **Fine-tuning Round 3 (Epochs 1-10)**: Performance stabilized around 73-74%

### Best Checkpoint Across All Training
**Fine-tuning Round 1, Epoch 10** (same as model_best.ckpt):
- R1@0.5: 74.71% (highest across all 81 epochs)
- R1@0.7: 60.77%
- mAP: 53.87%
- mAP@0.5: 73.25%
- mAP@0.75: 56.66%

---

## Key Findings

1. **Optimal Training Duration**: The model achieved peak performance after fine-tuning for 10 epochs from the best checkpoint at epoch 47 of the initial 50-epoch training.

2. **Consistent Performance**: The model shows stable performance across epochs 8-9 of fine-tuning, with minimal variance.

3. **Video Length Sensitivity**:
   - Strong performance on long videos (58.29% mAP)
   - Strong performance on middle-length videos (57.83% mAP)
   - Weak performance on short videos (14.79% mAP) - potential area for improvement

4. **Highlight Quality**: The model excels at detecting Fair and Good quality highlights but has room for improvement on VeryGood quality highlights.

5. **IoU Threshold Robustness**: The model maintains good performance across different IoU thresholds, with graceful degradation as the threshold increases.

---

## Files Generated

### Validation Set Results
- **Predictions**: `best_hl_val_preds.jsonl` (2.2M)
- **Metrics**: `best_hl_val_preds_metrics.json` (3.2K)
- **Evaluation Log**: `eval.log.txt` (19K)

### Checkpoint Files
- **Best Model**: `model_best.ckpt` (140M) - Epoch 9
- **Latest Model**: `model_latest.ckpt` (140M) - Epoch 9
- **Epoch 7 Checkpoint**: `model_e0007.ckpt` (140M)

---

## Next Steps

### For Testing on Other Datasets

To evaluate this checkpoint on other datasets (TACOS, Charades-STA), you would need to:

1. **Prepare the data format**: Ensure the dataset follows the same format as QVHighlights (with qid, query, vid fields)
2. **Update feature paths**: Point to the correct video and text feature directories
3. **Run inference**: Use the same inference script with the appropriate config

### Test Set Submission

To generate predictions for the QVHighlights test set (for official submission):
```bash
PYTHONPATH=$PYTHONPATH:. python sim_detr/inference.py \
    --resume results_internvideo2_finetune/hl-video_tef-finetune_from_epoch47_10ep-2025_12_11_20_54_10/model_best.ckpt \
    --eval_split_name test \
    --eval_path <path_to_test_jsonl>
```

**Note**: Test set evaluation requires the correct data format matching the training data structure.

---

## Conclusion

The Sim-DETR model with InternVideo2 features achieves strong performance on QVHighlights validation set, with **74.71% R1@0.5** and **53.87% mAP** for moment retrieval, and **81.96% Fair-mAP** for highlight detection. This represents the best performance achieved across all 81 training epochs.

The model is particularly effective on long and middle-length videos, with potential for improvement on short videos and very high-quality highlight detection.

---

**Report Generated**: 2025-12-11
**Model**: Sim-DETR with InternVideo2 features
**Dataset**: QVHighlights Validation Set
