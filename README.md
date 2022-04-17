# BayeSR
The official implementation of "[Bayesian Image Super-Resolution with Deep Modeling of Image Statistics](https://ieeexplore.ieee.org/document/9744488)" which has been accepted by *IEEE Transactions on Pattern Analysis and Machine Intelligence*.

## Content
- [BayeSR](#bayesr)
  - [Content](#content)
  - [Wiki](#wiki)
  - [Dependencies](#dependencies)
  - [Quick test](#quick-test)
    - [Ideal image super-resolution](#ideal-image-super-resolution)
    - [Realistic image super-resolution](#realistic-image-super-resolution)
    - [Real-world image super-resolution](#real-world-image-super-resolution)
    - [Synthetic image super-resolution](#synthetic-image-super-resolution)
  - [How to train](#how-to-train)
    - [Prepare training datasets](#prepare-training-datasets)
    - [Start training](#start-training)
  - [Citation](#citation)

## Wiki
Please refer to [wiki page](/../../wiki) for the detailed introduction of BayeSR.

## Dependencies
BayeSR was implemented on *Ubuntu 16.04* with *Python 3.6*. Before training and test, please create an environment via [Anaconda](https://www.anaconda.com/) (suppose it has been installed on your computer), and install tensorflow 1.10.0, as follows,
```bash
conda create -n BayeSR python=3.6
source activate BayeSR
conda install tensorflow-gpu==1.10.0
```
Besides, please install the following packages using ```pip install -r requirements.txt```.
- numpy==1.14.5
- opencv-python==4.4.0
- tqdm==4.51.0
- scikit-image==0.17.1
- Pillow==8.0.1
- scipy==1.2.1

## Quick test
BayeSR was tested on the widely used benchmark datasets for the tasks of ideal SISR x4, realistic SISR x4, and real-world SISR x4. We have provided the benchmark datasets and the pre-trained models via the following links and extraction codes.

|Datasets/Models|Upscale|Parameters|BaiduPan|OneDrive|
|-----|-------|----------|--------|--------|
|benchmark|   |          |[link](https://pan.baidu.com/s/1M9Mj2OPJRmd_aUszhaxFmA) `ggeh`|[link]()|
|downsamplors|x4  |0.15M      |[link](https://pan.baidu.com/s/1tRJyTizhpsiEHtbTS3vL1Q) `337e`|[link]()|
|IdealSISR|x4  |2.63M  |[link](https://pan.baidu.com/s/17gg2CQZeh1VslOZ98lT1fA) `8p47`|[link]()|
|RealisticSISR|x4  |2.63M  |[link](https://pan.baidu.com/s/1GM5bzojYtlbZy4HTtfmkHA) `jtjy`|[link]()|
|RealworldSISR|x4  |2.63M  |[link](https://pan.baidu.com/s/1VWf0Rj0eXf9AQHAutFn9pA) `qzj2`|[link]()|
|SyntheticSISR|x4  |2.63M  |[link](https://pan.baidu.com/s/1K1OyZE0JZqohRJzTp4qmxA) `77g7`|[link]()|

- `benchmark.zip` contains seven datasets:
  - `Set5`, `Set14`, `B100`, and `Urban100` for ideal SISR
  - `DIV2K_mild` for realistic SISR
  - `DPED_iphone` for real-world SISR
  - `RealWorld` for test in real-world scenarios.
- `downsamplors.zip` contains four downsampling models:
  - `BayeSR-Downsamplor-Bicubicx4` trained by supervised learning for ideal SISR x4
  - `BayeSR-Downsamplor-BiSRx4` trained by unsupervised [KernelGAN](https://www.wisdom.weizmann.ac.il/~vision/kernelgan/) for ideal SISR x4
  - `BayeSR-Downsamplor-ReSRx4` trained by unsupervised KernelGAN for realistic SISR x4
  - `BayeSR-Downsamplor-RWSRx4` trained by unsupervised KernelGAN for real-world SISR x4
- `IdealSISR.zip` contains two BayeSR models for ideal SISR x4:
  - `BayeSR-BiSRx4-sup` trained by the supervised strategy of BayeSR
  - `BayeSR-BiSRx4-unsup` trained by the unsupervised strategy of BayeSR
- `RealisticSISR.zip` contains three BayeSR models for realistic SISR x4:
  - `BayeSR-ReSRx4-sup` trained by the supervised strategy of BayeSR
  - `BayeSR-ReSRx4-psup` trained by the pseudo-supervised strategy of BayeSR
  - `BayeSR-ReSRx4-unsup` trained by the unsupervised strategy of BayeSR
- `RealworldSISR.zip` contains two BayeSR models for real-world SISR x4:
  - `BayeSR-RWSRx4-psup` trained by the pseudo-supervised strategy of BayeSR
  - `BayeSR-RWSRx4-unsup` trained by the unsupervised strategy of BayeSR
- `SyntheticSISR.zip` contains one BayeSR models:
  - `BayeSR-SySRx4-sup` trained by the supervised strategy of BayeSR, where LR images were corrupted by stochastic Gaussian kernel and Gaussian noise, instead of a pret-trained downsampling model. 

We have provided the script of testing BayeSR in `src/demo.sh`. Please `cd` your working path to `src/` and start test BayeSR as follows.
 
### Ideal image super-resolution
To test the generalizability of BayeSR on ideal SISR x4, please uncomment the following line in `demo.sh`, and then run `sh demo.sh`.
```bash
# supervised learning
python bayesr_test.py --dataset Set5 --input_data_dir ../data/TestDatasets/benchmark --task BiSR --sigma_read 20 --bayesr_checkpoint ../models/IdealSISR/BayeSR-BiSRx4-sup/model --GPU_ids 0
```
Here, `--sigma_read` defines the level of addictive white Gaussian noise. **Note that** the printed PSNR and SSIM values may be higher than the reported PSNR and SSIM in our article, since the latter were computed by a MATLAB script (`utils/PSNR_SSIM/PSNR_SSIM.m`) consistent with the previous works such as [SRCNN](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html) and [VDSR](https://github.com/huangzehao/caffe-vdsr).

To test the unsupervised performance of BayeSR on ideal SISR x4, please uncomment the following line in `demo.sh`, and then run `sh demo.sh`.
```bash
# unsupervised learning
python bayesr_test.py --dataset Set5 --input_data_dir ../data/TestDatasets/benchmark --task BiSR --bayesr_checkpoint ../models/IdealSISR/BayeSR-BiSRx4-unsup/model --GPU_ids 0
```

### Realistic image super-resolution
To test the supervised, pseudo-supervised, and unsupervised performance of BayeSR on realistic SISR x4, please uncomment one of the following lines in `demo.sh`, and then run `sh demo.sh`.
```bash
# supervised learning
python bayesr_test.py --dataset DIV2K_mild --input_data_dir ../data/TestDatasets/benchmark --task ReSR --bayesr_checkpoint ../models/RealisticSISR/BayeSR-ReSRx4-sup/model --GPU_ids 0
# pseudo-supervised learning
python bayesr_test.py --dataset DIV2K_mild --input_data_dir ../data/TestDatasets/benchmark --task ReSR --bayesr_checkpoint ../models/RealisticSISR/BayeSR-ReSRx4-psup/model --GPU_ids 0
# unsupervised learning
python bayesr_test.py --dataset DIV2K_mild --input_data_dir ../data/TestDatasets/benchmark --task ReSR --bayesr_checkpoint ../models/RealisticSISR/BayeSR-ReSRx4-unsup/model --GPU_ids 0
```

### Real-world image super-resolution
To test the pseudo-supervised and unsupervised performance of BayeSR, please uncomment one of the following lines in `demo.sh`, and then run `sh demo.sh`.
```bash
# pseudo-supervised learning
python bayesr_test.py --dataset DPED_iphone --input_data_dir ../data/TestDatasets/benchmark --task RWSR --bayesr_checkpoint ../models/RealworldSISR/BayeSR-RWSRx4-psup/model --GPU_ids 0
# unsupervised learning
python bayesr_test.py --dataset DPED_iphone --input_data_dir ../data/TestDatasets/benchmark --task RWSR --bayesr_checkpoint ../models/RealworldSISR/BayeSR-RWSRx4-unsup/model --GPU_ids 0
```

### Synthetic image super-resolution
To test the generalizability of BayeSR to diverse degradation kernels, please uncomment the following line in `demo.sh`, and then run `sh demo.sh`.
```bash
# supervised learning
python bayesr_kernels.py --dataset Set14 --input_data_dir ../data/TestDatasets/benchmark --task SySR --bayesr_checkpoint ../models/SyntheticSISR/BayeSR-SySRx4-sup/model --GPU_ids 0
```
Here, the default degradation kerenl is the kernel with respect to bicubic interpolation.

To test the performance of BayeSR in real-world scenarios, please uncomment the following line in `demo.sh`, and then run `sh demo.sh`.
```bash
# supervised learning
python bayesr_test.py --dataset RealWorld --input_data_dir ../data/TestDatasets/benchmark --task RWSR --bayesr_checkpoint ../models/SyntheticSISR/BayeSR-SySRx4-sup/model --GPU_ids 0
```
Here, the dataset `RealWorld` can be replaced with your own real-world image dataset.

## How to train
Before training, please prepare training datasets at first. Then, the downsampling module of BayeSR was pretrained by the unsupervised [KernelGAN](https://www.wisdom.weizmann.ac.il/~vision/kernelgan/). Finally, BayeSR was trained by fixing the downsampling model.
### Prepare training datasets
[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/), [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar), and [DPED](http://people.ee.ethz.ch/~ihnatova/index.html) were used to train BayeSR. All datasets should be included in the folder `data/TrainDatasets/` following a specific directory structure, namely,
```bash
DIV2K
  |--DIV2K_HR
       |--DIV2K_HR_train
       |--DIV2K_HR_valid
  |--DIV2K_LR_bicubic
       |--DIV2K_LR_train
            |--X4
       |--DIV2K_LR_valid
            |--X4
  |--DIV2K_LR_mild
       |--DIV2K_LR_train
            |--X4
            |--X4_noise
Flickr2K
  |--Flickr2K_HR
DPED
  |--iphone
       |--train_LR
       |--train_LR_noise
       |--valid_LR
```
Here, `*_noise` contains extracted noise patches from the LR image dataset `*`. For examples, `DPED/iphone/train_LR_noise` can be generated by `cd` to `data/` and running `python extract_nosie.py`.

### Start training
To pre-train downsampling models, please uncomment the following line in `demo.sh`, and run `sh demo.sh`.
```bash
# unsupervised learning
python bayesr_train.py --task RWSR --trained_net Downsamplor --train_type unsupervised --log_dir ../models/downsamplors/BayeSR-Downsamplor-RWSRx4 --GPU_ids 0
```
Here, `--task` can be set to BiSR, ReSR, and RWSR. The resulting downsampling models are corresponding to ideal SISR, realistic SISR, and real-world SISR. 

To train BayeSR on the task of ideal SISR x4, please uncomment one of the following line in `demo.sh`, and run `sh demo.sh`.
```bash
# supervised learning
python bayesr_train.py --task BiSR --train_type supervised --downsamplor_checkpoint ../models/downsamplors/BayeSR-Downsamplor-BiSRx4/model --log_dir ../models/IdealSISR/BayeSR-BiSRx4-sup --GPU_ids 0
# unsupervised learning
python bayesr_train.py --task BiSR --GAN_type GAN --train_type unsupervised --downsamplor_checkpoint ../models/downsamplors/BayeSR-Downsamplor-BiSRx4/model --log_dir ../models/IdealSISR/BayeSR-BiSRx4-unsup --GPU_ids 0
```
Here, `--downsamplor_checkpoint` points to the path of the pre-trained downsampling model for ideal SISR x4. `--GAN_type` defines the types of generative adversarial networks (GANs), and the choices are GAN, WGAN, and LSGAN.

To train BayeSR on the task of realistic SISR x4, please uncomment one of the following lines in `demo.sh`, and run `sh demo.sh`.
```bash
# supervised learning
python bayesr_train.py --task ReSR --train_type supervised --downsamplor_checkpoint ../models/downsamplors/BayeSR-Downsamplor-ReSRx4/model --log_dir ../models/RealisticSISR/BayeSR-ReSRx4-sup --GPU_ids 0
# pseudo-supervised learning
python bayesr_train.py --task ReSR --GAN_type GAN --train_type pseudosupervised --downsamplor_checkpoint ../models/downsamplors/BayeSR-Downsamplor-ReSRx4/model --log_dir ../models/RealisticSISR/BayeSR-ReSRx4-psup --GPU_ids 0
# unsupervised learning
python bayesr_train.py --task ReSR --GAN_type GAN --train_type unsupervised --downsamplor_checkpoint ../models/downsamplors/BayeSR-Downsamplor-ReSRx4/model --log_dir ../models/RealisticSISR/BayeSR-ReSRx4-unsup --GPU_ids 0
```

To train BayeSR on the task of real-world SISR x4, please uncommnet one of the following lines in `demo.sh`, and then run `sh demo.sh`.
```bash
# pseudo-supervised learning
python bayesr_train.py --task RWSR --GAN_type GAN --train_type pseudosupervised --downsamplor_checkpoint ../models/downsamplors/BayeSR-Downsamplor-RWSRx4/model --log_dir ../models/RealworldSISR/BayeSR-RWSRx4-psup --GPU_ids 0
# unsupervised learning
python bayesr_train.py --task RWSR --GAN_type GAN --train_type unsupervised --downsamplor_checkpoint ../models/downsamplors/BayeSR-Downsamplor-RWSRx4/model --log_dir ../models/RealworldSISR/BayeSR-RWSRx4-unsup --GPU_ids 0
```

To train BayeSR on the task of synthetic SISR x4, where LR images were degraded by random Gaussian kernel and Gaussian noise, please uncomment the following line in `demo.sh`, and then run `sh demo.sh`.
```bash
# supervised learning
python bayesr_train.py --task SySR --train_type supervised --gauss_blur --downsamplor_checkpoint ../models/downsamplors/BayeSR-Downsamplor-BiSRx4/model --log_dir ../models/SyntheticSISR/BayeSR-SySRx4-sup --GPU_ids 0
```

## Citation
If our work is helpful in your research, please cite this as follows.

[1] S. Gao and X. Zhuang, "Bayesian Image Super-Resolution with Deep Modeling of Image Statistics," in IEEE Transactions on Pattern Analysis and Machine Intelligence, doi: 10.1109/TPAMI.2022.3163307. [[arXiv]](https://arxiv.org/abs/2204.00623) [[TPAMI]](https://ieeexplore.ieee.org/document/9744488)
```
@ARTICLE{9744488,
  author={Gao, Shangqi and Zhuang, Xiahai},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Bayesian Image Super-Resolution with Deep Modeling of Image Statistics}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2022.3163307}}
```

Don't hesitate to contact us via [shqgao@163.com]() or [zxh@fudan.edu.cn](), if you have any questions.
