"""
Dataset loader for Sim-DETR with InternVideo2 features support.
Modified from start_end_dataset.py to support InternVideo2 features from SDST dataset.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import random
import logging
from os.path import join, exists
from utils.basic_utils import load_jsonl, l2_normalize_np_array
from utils.tensor_utils import pad_sequences_1d
from sim_detr.span_utils import span_xx_to_cxw

logger = logging.getLogger(__name__)


class StartEndDatasetInternVideo2(Dataset):
    """
    Dataset for QVHighlights with InternVideo2 features.

    InternVideo2 feature format:
    - Video features: .npz files with key 'visual_feats_interm_pooled', shape (L, K, D) = (75, 5, 768)
    - Text features: .npz files with key 'text_feats', shape (K, T, D) = (5, 17, 1024)

    Where:
    - L: video length (number of frames)
    - K: number of intermediate layers (5 for InternVideo2)
    - D: feature dimension (768 for video, 1024 for text)
    - T: text token length
    """

    Q_FEAT_TYPES = ["pooler_output", "last_hidden_state"]

    def __init__(self, dset_name, data_path, v_feat_dirs, q_feat_dir,
                 q_feat_type="last_hidden_state",
                 max_q_l=32, max_v_l=75, data_ratio=1.0, ctx_mode="video",
                 normalize_v=True, normalize_t=True, load_labels=True,
                 clip_len=2, max_windows=5, span_loss_type="l1", txt_drop_ratio=0,
                 dset_domain=None, use_internvideo2=False):
        self.dset_name = dset_name
        self.data_path = data_path
        self.data_ratio = data_ratio
        self.v_feat_dirs = v_feat_dirs if isinstance(v_feat_dirs, list) else [v_feat_dirs]
        self.q_feat_dir = q_feat_dir
        self.q_feat_type = q_feat_type
        self.max_q_l = max_q_l
        self.max_v_l = max_v_l
        self.ctx_mode = ctx_mode
        self.use_tef = "tef" in ctx_mode
        self.use_video = "video" in ctx_mode
        self.normalize_t = normalize_t
        self.normalize_v = normalize_v
        self.load_labels = load_labels
        self.clip_len = clip_len
        self.max_windows = max_windows
        self.span_loss_type = span_loss_type
        self.txt_drop_ratio = txt_drop_ratio
        self.use_internvideo2 = use_internvideo2

        if "val" in data_path or "test" in data_path:
            assert txt_drop_ratio == 0

        # checks
        assert q_feat_type in self.Q_FEAT_TYPES

        # data
        self.data = self.load_data()

    def load_data(self):
        datalist = load_jsonl(self.data_path)
        if self.data_ratio != 1:
            n_examples = int(len(datalist) * self.data_ratio)
            datalist = datalist[:n_examples]
            logger.info("Using {}% of the data: {} examples"
                        .format(self.data_ratio * 100, n_examples))
        return datalist

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        meta = self.data[index]

        model_inputs = dict()
        model_inputs["query_feat"] = self._get_query_feat_by_qid(meta["qid"])
        if self.use_video:
            model_inputs["video_feat"] = self._get_video_feat_by_vid(meta["vid"])
            ctx_l = len(model_inputs["video_feat"])
        else:
            ctx_l = self.max_v_l

        if self.use_tef:
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
            if self.use_video:
                model_inputs["video_feat"] = torch.cat(
                    [model_inputs["video_feat"], tef], dim=1)  # (Lv, Dv+2)
            else:
                model_inputs["video_feat"] = tef

        if self.load_labels:
            if 'relevant_clip_ids' in meta:
                pos_idx = torch.tensor(meta['relevant_clip_ids'])
                mask = torch.zeros_like(torch.ones(ctx_l))

                if pos_idx.max() >= len(mask):
                    new_mask = torch.zeros_like(torch.ones(pos_idx.max()+1))
                    new_mask[pos_idx] = 1
                    new_mask[:len(mask)] = mask
                    mask = new_mask
                else:
                    mask[pos_idx] = 1

                model_inputs["pos_mask"] = mask

            model_inputs["span_labels"] = self.get_span_labels(meta["relevant_windows"], ctx_l)
            if "relevant_clip_ids" in meta:
                model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
                    self.get_saliency_labels_all(meta["relevant_clip_ids"], meta["saliency_scores"], ctx_l)

        return dict(meta=meta, model_inputs=model_inputs)

    def get_saliency_labels_all(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """Sum the scores from the three annotations."""
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        score_array = np.zeros(ctx_l)
        for idx in range(len(rel_clip_ids)):
            if rel_clip_ids[idx] >= ctx_l:
                # Expand score_array to accommodate the larger index
                score_array_new = np.zeros(rel_clip_ids[idx] + 1)
                score_array_new[:len(score_array)] = score_array
                score_array = score_array_new
            score_array[rel_clip_ids[idx]] = agg_scores[idx]

        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices, score_array

    def get_span_labels(self, windows, ctx_l):
        """
        windows: list([st, ed]) in seconds.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        if len(windows) > self.max_windows:
            windows = windows[:self.max_windows]
        if self.span_loss_type == "l1":
            windows = torch.Tensor(windows) / (ctx_l * self.clip_len)  # normalized windows in xx
            windows = span_xx_to_cxw(windows)  # normalized windows in cxw
        elif self.span_loss_type == "ce":
            windows = torch.Tensor([
                [int(w[0] / self.clip_len), min(int(w[1] / self.clip_len), ctx_l) - 1]
                for w in windows]).long()  # inclusive
        else:
            raise NotImplementedError
        return windows

    def _get_query_feat_by_qid(self, qid):
        """Load text features from InternVideo2 format or original CLIP format."""
        q_feat_path = join(self.q_feat_dir, f"{qid}.npz" if self.use_internvideo2 else f"{qid}.npy")

        if self.use_internvideo2:
            # InternVideo2 format: .npz with key 'text_feats', shape (K, T, D)
            q_feat_data = np.load(q_feat_path)
            q_feat = q_feat_data['text_feats'].astype(np.float32)  # (K, T, D)
            # Reshape to (T, K*D) to match expected format
            K, T, D = q_feat.shape
            q_feat = q_feat.transpose(1, 0, 2)  # (T, K, D)
            q_feat = q_feat[:self.max_q_l]  # Limit to max_q_l
            if self.normalize_t:
                q_feat = l2_normalize_np_array(q_feat)
            if self.txt_drop_ratio > 0:
                q_feat = self.random_drop_rows(q_feat)
            l = len(q_feat)
            q_feat = q_feat.reshape(l, -1)  # (T, K*D)
        else:
            # Original CLIP format: .npy, shape (L, K, D)
            q_feat = np.load(q_feat_path).astype(np.float32)
            if self.q_feat_type == "last_hidden_state":
                q_feat = q_feat[:self.max_q_l]
            if self.normalize_t:
                q_feat = l2_normalize_np_array(q_feat)
            if self.txt_drop_ratio > 0:
                q_feat = self.random_drop_rows(q_feat)
            l = len(q_feat)
            q_feat = q_feat.reshape(l, -1)

        return torch.from_numpy(q_feat)

    def random_drop_rows(self, embeddings):
        """Randomly mask num_drop rows in embeddings to be zero."""
        num_drop_rows = round(len(embeddings) * self.txt_drop_ratio)
        if num_drop_rows > 0:
            row_indices = np.random.choice(
                len(embeddings), size=num_drop_rows, replace=False)
            embeddings[row_indices] = 0
        return embeddings

    def _get_video_feat_by_vid(self, vid):
        """Load video features from InternVideo2 format or original format."""
        v_feat_list = []

        for _feat_dir in self.v_feat_dirs:
            if self.use_internvideo2 or 'internvideo' in _feat_dir.lower():
                # InternVideo2 format: .npz with key 'visual_feats_interm_pooled', shape (L, K, D)
                _feat_path = join(_feat_dir, f"{vid}.npz")
                _feat_data = np.load(_feat_path)
                _feat = _feat_data['visual_feats_interm_pooled'][:self.max_v_l].astype(np.float32)  # (L, K, D)
                if self.normalize_v:
                    _feat = l2_normalize_np_array(_feat)
                l = len(_feat)
                _feat = _feat.reshape(l, -1)  # (L, K*D)
                v_feat_list.append(_feat)
            elif 'slowfast' in _feat_dir:
                _feat_path = join(_feat_dir, f"{vid}.npz")
                _feat = np.load(_feat_path)["features"][:self.max_v_l].astype(np.float32)
                if self.normalize_v:
                    _feat = l2_normalize_np_array(_feat)
                v_feat_list.append(_feat)
            elif 'clip' in _feat_dir:
                _feat_path = join(_feat_dir, f"{vid}.npy")
                _feat = np.load(_feat_path).astype(np.float32)  # L, K, D
                if self.normalize_v:
                    _feat = l2_normalize_np_array(_feat)
                l = len(_feat)
                _feat = _feat.reshape(l, -1)
                v_feat_list.append(_feat)

        # some features are slightly longer than the others
        min_len = min([len(e) for e in v_feat_list])
        v_feat_list = [e[:min_len] for e in v_feat_list]
        v_feat = np.concatenate(v_feat_list, axis=1)
        return torch.from_numpy(v_feat)  # (Lv, D)


def start_end_collate(batch):
    batch_meta = [e["meta"] for e in batch]

    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = dict()
    for k in model_inputs_keys:
        if k == "span_labels":
            batched_data[k] = [dict(spans=e["model_inputs"]["span_labels"]) for e in batch]
            continue
        if k in ["saliency_pos_labels", "saliency_neg_labels"]:
            batched_data[k] = torch.LongTensor([e["model_inputs"][k] for e in batch])
            continue
        if k == "saliency_all_labels":
            pad_data, mask_data = pad_sequences_1d([e["model_inputs"][k] for e in batch], dtype=np.float32, fixed_length=None)
            batched_data[k] = torch.tensor(pad_data, dtype=torch.float32)
            continue

        batched_data[k] = pad_sequences_1d(
            [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None)
    return batch_meta, batched_data


def prepare_batch_inputs(batched_model_inputs, device, non_blocking=False):
    model_inputs = dict(
        src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
        src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
        src_vid=batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
        src_vid_mask=batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
    )
    targets = {}
    if "span_labels" in batched_model_inputs:
        targets["span_labels"] = [
            dict(spans=e["spans"].to(device, non_blocking=non_blocking))
            for e in batched_model_inputs["span_labels"]
        ]
    if "saliency_pos_labels" in batched_model_inputs:
        for name in ["saliency_pos_labels", "saliency_neg_labels"]:
            targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)

    if "saliency_all_labels" in batched_model_inputs:
        targets["saliency_all_labels"] = batched_model_inputs["saliency_all_labels"].to(device, non_blocking=non_blocking)

    if "pos_mask" in batched_model_inputs:
        targets['src_pos_mask']=batched_model_inputs["pos_mask"][0].to(device, non_blocking=non_blocking)
    targets = None if len(targets) == 0 else targets
    return model_inputs, targets
