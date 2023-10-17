# Regress Before Construct: Regress Autoencoder for Point Cloud Self-supervised Learning
 
[OpenReview](https://openreview.net/forum?id=ePXxwgvwJ9) | [arXiv](https://arxiv.org/pdf/2310.03670.pdf) 

This repository contains the code release of paper **Regress Before Construct: Regress Autoencoder for Point Cloud Self-supervised Learning** (ACM MM 2023).

## Abstract 

Masked Autoencoders (MAE) have demonstrated promising performance in self-supervised learning for both 2D and 3D computer vision. Nevertheless, existing MAE-based methods still have certain drawbacks. Firstly, the functional decoupling between the encoder and decoder is incomplete, which limits the encoder's representation learning ability. Secondly, downstream tasks solely utilize the encoder, failing to fully leverage the knowledge acquired through the encoder-decoder architecture in the pre-text task. In this paper, we propose Point Regress AutoEncoder (Point-RAE), a new scheme for regressive autoencoders for point cloud self-supervised learning.  The proposed method decouples functions between the decoder and the encoder by introducing a mask regressor, which predicts the masked patch representation from the visible patch representation encoded by the encoder and the decoder reconstructs the target from the predicted masked patch representation. By doing so, we minimize the impact of decoder updates on the representation space of the encoder. Moreover, we introduce an alignment constraint to ensure that the representations for masked patches, predicted from the encoded representations of visible patches, are aligned with the masked patch presentations computed from the encoder. To make full use of the knowledge learned in the pre-training stage, we design a new finetune mode for the proposed Point-RAE. Extensive experiments demonstrate that our approach is efficient during pre-training and generalizes well on various downstream tasks. Specifically, our pre-trained models achieve a high accuracy of 90.28% on the ScanObjectNN hardest split and 94.1% accuracy on ModelNet40, surpassing all the other self-supervised learning methods. 
<div  align="center">    
 <img src="./figure/framework.png" width = "666"  align=center />
</div>


## Environment

This codebase was tested with the following environment configurations. It may work with other versions.
- Ubuntu 18.04
- CUDA 11.3
- GCC 7.5.0
- Python 3.8.8
- PyTorch 1.10.0

## 1. Installation
We recommend using Anaconda for the installation process:
```shell
# Make sure `g++-7 --version` is at least 7.4.0
$ sudo apt install g++-7  # For CUDA 10.2, must use GCC < 8

# Create virtual env and install PyTorch
$ conda create -n act python=3.8.8
$ conda activate act

(act) $ conda install openblas-devel -c anaconda
(act) $ conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -c nvidia
# Or, you can set up Pytorch with pip from official link:
# (act) $ pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html # recommended
# For CUDA 10.2, use conda:
# (act) $ conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch -c nvidia
# Or pip:
# (act) $ pip install torch==1.11.0+cu102 torchvision==0.12.0+cu102 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu102

# Install basic required packages
(act) $ pip install -r requirements.txt

# Chamfer Distance
(act) $ cd ./extensions/chamfer_dist && python setup.py install --user
# PointNet++
(act) $ pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
(act) $ pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## 2. Datasets

We use ShapeNet, ScanObjectNN, ModelNet40, S3DIS and ShapeNetPart in this work. See [DATASET.md](./DATASET.md) for details.

## 3. Models

We will release all the pretrained models soon


## 4. Point-RAE Pre-training
To pretrain Point-RAE on the ShapeNet training set, run the following command. If you want to try different models or masking ratios etc., first create a new config file, and pass its path to --config.

Point-RAE pretraining includes two stages:

* Stage I, we follow ACT to transfer pretrained Transformer on ShapeNet as 3D autoencoder by running:

  ```shell
  CUDA_VISIBLE_DEVICES=<GPUs> python main_autoencoder.py \
      --config "cfgs/autoencoder/act_dvae_with_pretrained_transformer.yaml" \
      --exp_name <output_file_name>
  ```

  or

  ```shell
  sh train_autoencoder.sh <GPU>
  ```

* Stage II, pretrain Point-RAE on ShapeNet by running:

  ```shell
  CUDA_VISIBLE_DEVICES=<GPUs> \
      python main.py --config "cfgs/pretrain/pretrain.yaml" \
      --exp_name <output_file_name>
  ```

  or

  ```shell
  sh pretrain.sh <GPU>
  ```

## 5. ACT Fine-tuning

Fine-tuning on ScanObjectNN, run:
```shell
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/finetune_classification/full/finetune_scan_hardest.yaml \
--finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
```
Fine-tuning on ModelNet40, run:
```shell
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/finetune_classification/full/finetune_modelnet.yaml \
--finetune_model --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
```
Voting on ModelNet40, run:
```shell
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --test --config cfgs/finetune_classification/full/finetune_modelnet.yaml \
--exp_name <output_file_name> --ckpts <path/to/best/fine-tuned/model>
```
Few-shot learning, run:
```shell
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/finetune_classification/few_shot/fewshot_modelnet.yaml --finetune_model \
--ckpts <path/to/pre-trained/model> --exp_name <output_file_name> --way <5 or 10> --shot <10 or 20> --fold <0-9>
```


## License
ACT is released under the MIT License. See the [LICENSE](./LICENSE) file for more details. Besides, the licensing information for `pointnet2` modules is available [here](https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/UNLICENSE).

## Acknowledgements
Many thanks to the following codes that help us a lot in building this codebase:
* [Point-BERT](https://github.com/lulutang0608/Point-BERT)
* [Point-MAE](https://github.com/Pang-Yatian/Point-MAE)
* [ACT](https://github.com/RunpeiDong/ACT)
* [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch)

## Contact

If you have any questions related to the code or the paper, feel free to email Runpei (`runpei.dong@gmail.com`). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation

If you find our work useful in your research, please consider citing:
```latex
@inproceedings{dong2023act,
  title={Autoencoders as Cross-Modal Teachers: Can Pretrained 2D Image Transformers Help 3D Representation Learning?},
  author={Runpei Dong and Zekun Qi and Linfeng Zhang and Junbo Zhang and Jianjian Sun and Zheng Ge and Li Yi and Kaisheng Ma},
  booktitle={The Eleventh International Conference on Learning Representations (ICLR) },
  year={2023},
  url={https://openreview.net/forum?id=8Oun8ZUVe8N}
}
```
