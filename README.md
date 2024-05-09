
# C<sup>2</sup>Former: Calibrated and Complementary Transformer for RGB-Infrared Object Detection

This repo is the official implementation for **C<sup>2</sup>Former: Calibrated and Complementary Transformer for RGB-Infrared Object Detection**. The paper has been accepted to **TGRS**.

## News
[2024.05.09] **Code** released!

[2024.04.15] Our paper is **accepted**!

## Overview
### Framework
<p align="center">
  <img src="README.assets\framework.png" alt="overview" width="90%">
</p>



### Visualization
<p align="center">
  <img src="README.assets\detection.png" alt="overview" width="90%">
</p>

## Results
[Kaist Results](https://drive.google.com/drive/folders/1a8Swf5BSgSyE6S6XdGuKhoygV_AyrDQ4?usp=sharing)

## Installation 

**Please refer to [[mmrotate installation](https://mmdetection.readthedocs.io/en/latest/get_started.html)] for installation.**

## Getting Started

### Train with a single GPU. 
```shell
python tools/train.py configs/s2anet/s2anet_c2former_fpn_1x_dota_le135.py --work-dir work_dirs/C2Former
```

### Inference

```shell
python tools/test.py configs/s2anet/s2anet_c2former_fpn_1x_dota_le135.py work_dirs/C2Former/${CHECKPOINT_FILE} --out work_dirs/C2Former/results.pkl
```


## Citation

If this is useful for your research, please consider cite.

```
@article{yuan2024c,
  title={C 2 Former: Calibrated and Complementary Transformer for RGB-Infrared Object Detection},
  author={Yuan, Maoxun and Wei, Xingxing},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2024},
  publisher={IEEE}
}

@inproceedings{yuan2022translation,
  title={Translation, scale and rotation: cross-modal alignment meets RGB-infrared vehicle detection},
  author={Yuan, Maoxun and Wang, Yinyan and Wei, Xingxing},
  booktitle={European Conference on Computer Vision},
  pages={509--525},
  year={2022},
  organization={Springer}
}
```