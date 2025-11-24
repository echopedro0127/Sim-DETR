# [ICCV2025] Sim-DETR: Unlock DETR for Temporal Sentence Grounding

by Jiajin Tang*, Zhengxuan Wei*, Yuchen Zhu, Cheng Shi, Guanbin Li, Liang Lin, Sibei Yang†

*Equal contribution; †Corresponding Author


[![arXiv:2509.23867](https://img.shields.io/badge/arXiv-2509.23867-red)](https://arxiv.org/abs/2509.23867)

----------

## Prerequisites

### 0. Clone this repository

```
git clone https://github.com/SooLab/Sim-DETR.git
cd Sim-DETR
```

### 1. Prepare datasets
#### QVHighlights

We use video features (CLIP and SlowFast) and text features (CLIP) as inputs. For CLIP, we utilize the features extracted by [R2-Tuning](https://github.com/yeliudev/R2-Tuning) (from the last four layers), but we retain only the `[CLS]` token per frame to ensure efficiency. You can download our prepared feature files from [qvhighlights\_features](https://drive.google.com/drive/folders/1rRVID6OO5arVR1vL5SP5fcCFAJ35B-IK?usp=sharing) and unzip them to your data root directory.


### 2. Install dependencies

For Anaconda setup, refer to the official [Moment-DETR GitHub](https://github.com/jayleicn/moment_detr).

----------

## QVHighlights

### Training

Update `feat_root` in `sim_detr/scripts/train.sh` to the path where you saved the features, then run:

```bash
bash sim_detr/scripts/train.sh  
```


### Inference Evaluation and Codalab Submission

After training, you can generate `hl_val_submission.jsonl` and `hl_test_submission.jsonl` for validation and test sets by running:

```
bash sim_detr/scripts/inference.sh results/{direc}/model_best.ckpt 'val'
bash sim_detr/scripts/inference.sh results/{direc}/model_best.ckpt 'test'
```

Replace `{direc}` with the path to your saved checkpoint. For more details on submission, see [standalone_eval/README.md](standalone_eval/README.md).

----------

## Citation

If you find this repository useful, please cite our work:

```
@inproceedings{tang2025sim,
  title={Sim-DETR: Unlock DETR for Temporal Sentence Grounding},
  author={Tang, Jiajin and Wei, Zhengxuan and Zhu, Yuchen and Shi, Cheng and Li, Guanbin and Lin, Liang and Yang, Sibei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={22760--22771},
  year={2025}
}
```

----------

## License

The annotation files and parts of the implementation are borrowed from Moment-DETR and TR-DETR. Consequently, our code is also released under the [MIT License](https://opensource.org/licenses/MIT).


