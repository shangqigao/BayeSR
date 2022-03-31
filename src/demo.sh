#!/bin/sh

#-----------------------BayeSR------------------------------#
#==================Downsamplor=============================#
# training example for downsampling operation
#python bayesr_train.py --task RWSR --trained_net Downsamplor --train_type unsupervised --log_dir ../models/downsamplors/BayeSR-Downsamplor-RWSRx4 --GPU_ids 0

#==================Ideal SISR x4==========================#
# training examples for ideal SISRx4
# supervised training
#python bayesr_train.py --task BiSR --train_type supervised --downsamplor_checkpoint ../models/downsamplors/BayeSR-Downsamplor-BiSRx4/model --log_dir ../models/IdealSISR/BayeSR-BiSRx4-sup --GPU_ids 0

# unsupervised training
#python bayesr_train.py --task BiSR --GAN_type GAN --train_type unsupervised --downsamplor_checkpoint ../models/downsamplors/BayeSR-Downsamplor-BiSRx4/model --log_dir ../models/IdealSISR/BayeSR-BiSRx4-unsup --GPU_ids 0

# test examples for ideal SISRx4
#python bayesr_test.py --dataset Set5 --input_data_dir ../data/TestDatasets/benchmark --task BiSR --sigma_read 20 --bayesr_checkpoint ../models/IdealSISR/BayeSR-BiSRx4-sup/model --GPU_ids 0

#python bayesr_test.py --dataset Set5 --input_data_dir ../data/TestDatasets/benchmark --task BiSR --bayesr_checkpoint ../models/IdealSISR/BayeSR-BiSRx4-unsup/model --GPU_ids 0

#=================Realistic SISR x4========================#
# training examples for realistic SISRx4
# supervised learning
#python bayesr_train.py --task ReSR --train_type supervised --downsamplor_checkpoint ../models/downsamplors/BayeSR-Downsamplor-ReSRx4/model --log_dir ../models/RealisticSISR/BayeSR-ReSRx4-sup --GPU_ids 0

# pseudo-supervised learning
#python bayesr_train.py --task ReSR --GAN_type GAN --train_type pseudosupervised --downsamplor_checkpoint ../models/downsamplors/BayeSR-Downsamplor-ReSRx4/model --log_dir ../models/RealisticSISR/BayeSR-ReSRx4-psup --GPU_ids 0

# unsupervised learning
#python bayesr_train.py --task ReSR --GAN_type GAN --train_type unsupervised --downsamplor_checkpoint ../models/downsamplors/BayeSR-Downsamplor-ReSRx4/model --log_dir ../models/RealisticSISR/BayeSR-ReSRx4-unsup --GPU_ids 0

# test examples for realistic SISRx4
#python bayesr_test.py --dataset DIV2K_mild --input_data_dir ../data/TestDatasets/benchmark --task ReSR --bayesr_checkpoint ../models/RealisticSISR/BayeSR-ReSRx4-sup/model --GPU_ids 0

#python bayesr_test.py --dataset DIV2K_mild --input_data_dir ../data/TestDatasets/benchmark --task ReSR --bayesr_checkpoint ../models/RealisticSISR/BayeSR-ReSRx4-psup/model --GPU_ids 0

#python bayesr_test.py --dataset DIV2K_mild --input_data_dir ../data/TestDatasets/benchmark --task ReSR --bayesr_checkpoint ../models/RealisticSISR/BayeSR-ReSRx4-unsup/model --GPU_ids 0

#=================Real-world SISR x4========================#
# training examples for real-world SISRx4
# pseudo-supervised learning
#python bayesr_train.py --task RWSR --GAN_type GAN --train_type pseudosupervised --downsamplor_checkpoint ../models/downsamplors/BayeSR-Downsamplor-RWSRx4/model --log_dir ../models/RealworldSISR/BayeSR-RWSRx4-psup --GPU_ids 0

# unsupervised learning
#python bayesr_train.py --task RWSR --GAN_type GAN --train_type unsupervised --downsamplor_checkpoint ../models/downsamplors/BayeSR-Downsamplor-RWSRx4/model --log_dir ../models/RealworldSISR/BayeSR-RWSRx4-unsup --GPU_ids 0

# test examples for real-world SISRx4
#python bayesr_test.py --dataset DPED_iphone --input_data_dir ../data/TestDatasets/benchmark --task RWSR --bayesr_checkpoint ../models/RealworldSISR/BayeSR-RWSRx4-psup/model --GPU_ids 0

#python bayesr_test.py --dataset DPED_iphone --input_data_dir ../data/TestDatasets/benchmark --task RWSR --bayesr_checkpoint ../models/RealworldSISR/BayeSR-RWSRx4-unsup/model --GPU_ids 0

#=================Synthetic SISR x4: generalizability to diverse kernels=========================#
# training examples for synthetic SISRx4 with random gaussian blur
# supervised learning
#python bayesr_train.py --task SySR --train_type supervised --gauss_blur --downsamplor_checkpoint ../models/downsamplors/BayeSR-Downsamplor-BiSRx4/model --log_dir ../models/SyntheticSISR/BayeSR-SySRx4-sup --GPU_ids 0

# test examples for synthetic SISRx4
# default kernel is the bicubic interpolation kernel
#python bayesr_kernels.py --dataset Set14 --input_data_dir ../data/TestDatasets/benchmark --task SySR --bayesr_checkpoint ../models/SyntheticSISR/BayeSR-SySRx4-sup/model --GPU_ids 0

#python bayesr_test.py --dataset RealWorld --input_data_dir ../data/TestDatasets/benchmark --task RWSR --bayesr_checkpoint ../models/SyntheticSISR/BayeSR-SySRx4-sup/model --GPU_ids 0
