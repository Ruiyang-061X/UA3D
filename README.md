# Harnessing Uncertainty-aware Bounding Boxes for Unsupervised 3D Object Detection

by [Ruiyang Zhang](https://ruiyang-061x.github.io/), Hu Zhang, Hang Yu, [Zhedong Zheng](https://www.zdzheng.xyz/)

## Motivation
![](.asset/Motivation.png)

## Abstract
Unsupervised 3D object detection aims to identify objects of interest from unla-beled raw data, such as LiDAR points. Recent approaches usually adopt pseudo 3D bounding boxes (3D bboxes) from clustering algorithm to initialize the model training, and then iteratively updating both pseudo labels and the trained model. However, pseudo bboxes inevitably contain noises, and such inaccurate annotation accumulates to the final model, compromising the performance. Therefore, in an attempt to mitigate the negative impact of pseudo bboxes, we introduce a new uncertainty-aware framework. In particular, Our method consists of two primary components: uncertainty estimation and uncertainty regularization. (1) In the uncertainty estimation phase, we incorporate an extra auxiliary detection branch alongside the primary detector. The prediction disparity between the primary and auxiliary detectors is leveraged to estimate uncertainty at the box coordinate level, including position, shape, orientation. (2) Based on the assessed uncertainty, we regularize the model training via adaptively adjusting every 3D bboxes coordinates. For pseudo bbox coordinates with high uncertainty, we assign a relatively low loss weight. Experiment verifies that the proposed method is robust against the noisy pseudo bboxes, yielding substantial improvements on
nuScenes compared to existing techniques, with increases of 6.9% in AP<sub>BEV</sub> and 2.5% in AP<sub>3D</sub> on nuScenes.

## Environment
- Follow [`MODEST/README.md`](https://github.com/YurongYou/MODEST/blob/master/README.md) to install.

## Uncertainty Learning for Unsupervised 3D Object Detection (UL3D)
### nuScenes (default PRCNN model)
```bash
conda activate UL3D; bash scripts/seed_training_nuscenes.sh; bash scripts/self_training_nusc.sh -C "data_paths=nusc.yaml det_filtering.pp_score_threshold=0.7 det_filtering.pp_score_percentile=20 data_paths.bbox_info_save_dst=null calib_path=$(pwd)/downstream/OpenPCDet/data/nuscenes_boston/training/calib ptc_path=$(pwd)/downstream/OpenPCDet/data/nuscenes_boston/training/velodyne image_shape=[900,1600]"
```
### Lyft (default PRCNN model)
```bash
conda activate UL3D; bash scripts/seed_training_lyft.sh; bash scripts/self_training_lyft.sh -C "det_filtering.pp_score_threshold=0.7 det_filtering.pp_score_percentile=20 data_paths.bbox_info_save_dst=null data_root=$(pwd)/downstream/OpenPCDet/data/lyft/training";
```

## Evaluation
### nuScenes (default PRCNN model)
```bash
conda activate UL3D; cd downstream/OpenPCDet/tools; bash scripts/dist_test.sh 4 --cfg_file ../../downstream/OpenPCDet/tools/cfgs/nuscenes_boston_models/pointrcnn_dynamic_obj.yaml --ckpt PATH_TO_CKPT
```
### Lyft (default PRCNN model)
```bash
conda activate UL3D; cd downstream/OpenPCDet/tools; bash scripts/dist_test.sh 4 --cfg_file ../../downstream/OpenPCDet/tools/cfgs/lyft_models/pointrcnn_dynamic_obj.yaml --ckpt PATH_TO_CKPT
```

## Checkpoints
### nuScenes experiments
| Model | ST rounds | Checkpoint  | Config file |
| ----- | :----:  | :----: | :----: |
| PointRCNN | 0 | [link](https://drive.google.com/file/d/1HrmG_QlJT_6ztN0NmqCMLpKRXhvduqj4/view?usp=sharing) | [cfg](downstream/OpenPCDet/tools/cfgs/nuscenes_boston_models/pointrcnn_dynamic_obj.yaml) |
| PointRCNN | 1 | [link](https://drive.google.com/file/d/13MkDu0p2_KEDKcOzHJxr3zieGaKO77TF/view?usp=sharing) | [cfg](downstream/OpenPCDet/tools/cfgs/nuscenes_boston_models/pointrcnn_dynamic_obj.yaml) |
| PointRCNN | 10 | [link](https://drive.google.com/file/d/12XM2oH7NxLkLS5omxjD4BMLlpI94heAl/view?usp=sharing) | [cfg](downstream/OpenPCDet/tools/cfgs/nuscenes_boston_models/pointrcnn_dynamic_obj.yaml) |

## Core Codes
- [`downstream/OpenPCDet/pcdet/models/backbones_3d/pointnet2_backbone.py`](downstream/OpenPCDet/pcdet/models/backbones_3d/pointnet2_backbone.py): Add auxiliary detector into original detection backbone.
- [`downstream/OpenPCDet/pcdet/models/dense_heads/point_head_template.py`](downstream/OpenPCDet/pcdet/models/dense_heads/point_head_template.py): Implement fine-grained uncertainty estimation and uncertainty regularization.
- [`downstream/OpenPCDet/pcdet/models/dense_heads/point_head_box.py`](downstream/OpenPCDet/pcdet/models/dense_heads/point_head_box.py): Necessary uncertainty-related params passage.
- [`downstream/OpenPCDet/pcdet/models/detectors/point_rcnn.py`](downstream/OpenPCDet/pcdet/models/detectors/point_rcnn.py): Combine primary detector loss and auxiliary detector loss into final supervision loss.
- [`downstream/OpenPCDet/tools/cfgs/nuscenes_boston_models/pointrcnn_dynamic_obj.yaml`](downstream/OpenPCDet/tools/cfgs/nuscenes_boston_models/pointrcnn_dynamic_obj.yaml): nuScenes hyperparameters for auxilary detector, including FPLayers, PointHead, and ROIHead.
- [`downstream/OpenPCDet/tools/cfgs/lyft_models/pointrcnn_dynamic_obj.yaml`](downstream/OpenPCDet/tools/cfgs/lyft_models/pointrcnn_dynamic_obj.yaml): Lyft hyperparameters for auxilary detector, including FPLayers, PointHead, and ROIHead.

## License
This project is under the MIT License.

## Contact
Please open an issue if you have any questions about using this repo.

## Acknowledgement
Our repo is based on [MODEST(CVPR'22)](https://github.com/YurongYou/MODEST), [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). Thanks for their great works and open-source effort!

## Citation
```bib
@inproceedings{zhang2024harnessing,
  title={Harnessing Uncertainty-aware Bounding Boxes for Unsupervised 3D Object Detection},
  author={Zhang, Ruiyang and Zhang, Hu and Yu, Hang and Zheng, Zhedong},
  booktitle={Arxiv},
  year={2024}
}
```