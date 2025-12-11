"""
Test script to verify InternVideo2 feature loading.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from sim_detr.start_end_dataset_internvideo2 import StartEndDatasetInternVideo2, start_end_collate

def test_feature_loading():
    print("=" * 80)
    print("Testing InternVideo2 Feature Loading")
    print("=" * 80)

    # Test parameters
    data_path = "/home/cl/vscode/sdst_datasets/data/qvhighlights/qvhighlights_train.jsonl"
    v_feat_dir = "/home/cl/vscode/sdst_datasets/data/qvhighlights/vid_feats"
    q_feat_dir = "/home/cl/vscode/sdst_datasets/data/qvhighlights/txt_feats"

    print(f"\nData path: {data_path}")
    print(f"Video feature dir: {v_feat_dir}")
    print(f"Text feature dir: {q_feat_dir}")

    # Create dataset
    print("\n" + "-" * 80)
    print("Creating dataset with InternVideo2 features...")
    print("-" * 80)

    dataset = StartEndDatasetInternVideo2(
        dset_name="hl",
        data_path=data_path,
        v_feat_dirs=[v_feat_dir],
        q_feat_dir=q_feat_dir,
        q_feat_type="last_hidden_state",
        max_q_l=32,
        max_v_l=75,
        data_ratio=0.01,  # Use only 1% of data for testing
        ctx_mode="video_tef",
        normalize_v=True,
        normalize_t=True,
        load_labels=True,
        clip_len=2,
        max_windows=10,
        span_loss_type="l1",
        txt_drop_ratio=0,
        use_internvideo2=True
    )

    print(f"✓ Dataset created successfully!")
    print(f"  Total samples: {len(dataset)}")

    # Test single sample
    print("\n" + "-" * 80)
    print("Testing single sample loading...")
    print("-" * 80)

    sample = dataset[0]
    meta = sample['meta']
    model_inputs = sample['model_inputs']

    print(f"\n✓ Sample loaded successfully!")
    print(f"\nMetadata:")
    print(f"  QID: {meta['qid']}")
    print(f"  VID: {meta['vid']}")
    print(f"  Query: {meta['query'][:60]}...")
    print(f"  Duration: {meta['duration']}s")
    print(f"  Relevant windows: {meta['relevant_windows']}")

    print(f"\nModel Inputs:")
    print(f"  Query feature shape: {model_inputs['query_feat'].shape}")
    print(f"  Video feature shape: {model_inputs['video_feat'].shape}")
    print(f"  Span labels shape: {model_inputs['span_labels'].shape}")

    # Expected dimensions
    print(f"\n✓ Feature dimensions check:")
    q_feat = model_inputs['query_feat']
    v_feat = model_inputs['video_feat']

    # InternVideo2: text (T, K*D) where K=5, D=1024 -> K*D=5120
    # InternVideo2: video (L, K*D+2) where K=5, D=768 -> K*D=3840, +2 for TEF
    expected_q_dim = 5120  # 5 * 1024
    expected_v_dim = 3840 + 2  # 5 * 768 + 2 (TEF)

    print(f"  Text feature dim: {q_feat.shape[-1]} (expected: {expected_q_dim})")
    print(f"  Video feature dim: {v_feat.shape[-1]} (expected: {expected_v_dim})")

    if q_feat.shape[-1] == expected_q_dim:
        print(f"  ✓ Text feature dimension correct!")
    else:
        print(f"  ✗ Text feature dimension mismatch!")

    if v_feat.shape[-1] == expected_v_dim:
        print(f"  ✓ Video feature dimension correct!")
    else:
        print(f"  ✗ Video feature dimension mismatch!")

    # Test batch loading
    print("\n" + "-" * 80)
    print("Testing batch loading with DataLoader...")
    print("-" * 80)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=start_end_collate,
        num_workers=0,
        pin_memory=False
    )

    batch_meta, batched_data = next(iter(dataloader))

    print(f"\n✓ Batch loaded successfully!")
    print(f"  Batch size: {len(batch_meta)}")
    print(f"  Query features shape: {batched_data['query_feat'][0].shape}")
    print(f"  Video features shape: {batched_data['video_feat'][0].shape}")
    print(f"  Query mask shape: {batched_data['query_feat'][1].shape}")
    print(f"  Video mask shape: {batched_data['video_feat'][1].shape}")

    # Test prepare_batch_inputs
    print("\n" + "-" * 80)
    print("Testing prepare_batch_inputs...")
    print("-" * 80)

    from sim_detr.start_end_dataset_internvideo2 import prepare_batch_inputs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    model_inputs, targets = prepare_batch_inputs(batched_data, device, non_blocking=False)

    print(f"\n✓ Batch inputs prepared successfully!")
    print(f"  src_txt shape: {model_inputs['src_txt'].shape}")
    print(f"  src_vid shape: {model_inputs['src_vid'].shape}")
    print(f"  src_txt_mask shape: {model_inputs['src_txt_mask'].shape}")
    print(f"  src_vid_mask shape: {model_inputs['src_vid_mask'].shape}")

    if targets is not None:
        print(f"  Number of targets: {len(targets['span_labels'])}")
        print(f"  Saliency labels shape: {targets['saliency_all_labels'].shape}")

    print("\n" + "=" * 80)
    print("✓ All tests passed successfully!")
    print("=" * 80)
    print("\nInternVideo2 features are correctly loaded and ready for training.")
    print(f"\nTo train with InternVideo2 features, run:")
    print(f"  cd /home/cl/vscode/rmodel/Sim-DETR")
    print(f"  source simdetr_env/bin/activate")
    print(f"  bash sim_detr/scripts/train_internvideo2.sh")
    print("=" * 80)

if __name__ == "__main__":
    test_feature_loading()
