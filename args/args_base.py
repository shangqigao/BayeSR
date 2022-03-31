
from argparse import ArgumentParser

def add_dataset_args(parser: ArgumentParser) -> None:
    parser.add_argument('--input_data_dir', type=str,
                        default='data/TrainDatasets',
                        help='Directory of the input data'
                        )
    parser.add_argument('--dataset', type=str, default='Set5',
                        help='Test dataset'
                        )
    parser.add_argument('--sample_num', type=int, default=-1,
                        help='which image is used to train, -1 denotes using all data'
                        )
    parser.add_argument('--augment', type=bool, default=True,
                        help='whether to augment data'
                        )
    parser.add_argument('--gauss_blur', action='store_true',
                        help='whether to degrade by random gaussian kernel'
                        )
    parser.add_argument('--motion_blur', action='store_true',
                        help='whether to degrade by random motion blur kernel'
                        )
    parser.add_argument('--sigma_read', type=float, default=0,
                        help='image read noise level, ranges in [0, 25]'
                        )
    parser.add_argument('--sigma_shot', type=float, default=0,
                        help='image shot noise level, ranges in [0, 8]'
                        )
    parser.add_argument('--range', type=bool, default=True,
                        help='whether let the noise level range in [0, sigma]'
                        )
    parser.add_argument('--train_batch_size', type=int, default=4,
                        help='Train batch size. Must divide evenly into dataset size'
                        )
    parser.add_argument('--valid_batch_size', type=int, default=16,
                        help='validation batch size. Must divide evenly into dataset size'
                        )
    parser.add_argument('--train_patch_size', type=int, default=32,
                        help='The batch size for training'
                        )
    parser.add_argument('--valid_patch_size', type=int, default=32,
                        help='The batch size for validation'
                        )
    parser.add_argument('--kernel_size', type=int, default=25,
                        help='The size of blur kernel'
                        )
    parser.add_argument('--input_kernel_dir', type=str, default=None,
                        help='Directory of kernels'
                        )
    parser.add_argument('--input_gt_kernel_dir', type=str, default=None,
                        help='Directory of true kernels'
                        )

def add_experiment_args(parser: ArgumentParser) -> None:
    parser.add_argument('--task', choices=['DEN', 'BiSR', 'SySR', 'ReSR', 'RWSR'], required=True,
                        help='Mode of task'
                        )
    parser.add_argument('--train_type', choices=['supervised', 'pseudosupervised', 'unsupervised'],
                        default='unsupervised',
                        help='which kind of training strategies to use'
                        )
    parser.add_argument('--upscale', type=int, default=4,
                        help='upscaling factor'
                        )
    parser.add_argument('--max_steps', type=int, default=1000000,
                        help='Maximal training steps'
                        )
    parser.add_argument('--regularizer', choices=['l2', 'l1'],
                        help='Which kind of regularizer to use'
                        )
    parser.add_argument('--log_dir', type=str,
                        default='./logs/BayeSR',
                        help='Directory to put the log data'
                        )
    parser.add_argument('--GPU_ids', type=str, default = '0',
                        help = 'Ids of GPUs'
                        )
    parser.add_argument('--resume', action='store_true',
                        help = 'If set, resume the training from a previous model checkpoint'
                        )
    parser.add_argument('--threads', type=int, default=2,
                        help='Number of threads for loading data'
                        )
    parser.add_argument('--save_dir', type=str, default='../results',
                        help='Directory of saving reconstructions'
                        )
    parser.add_argument('--ensemble', action='store_true',
                        help='If set, use data ensemble'
                        )

    
def add_model_args(parser: ArgumentParser) -> None:
    parser.add_argument('--trained_net', choices=['Downsamplor', 'BayeSR'],
                        default='BayeSR',
                        help='which network to train'
                        )
    parser.add_argument('--GAN_type', choices=['GAN', 'LSGAN', 'WGAN', None], 
                        default=None,
                        help='which type of GAN to use, if none, do not train GAN'
                        )
    parser.add_argument('--in_channel', type=int, default=3,
                        help='The number of input channels'
                        )
    parser.add_argument('--filters', type=int, default=64,
                        help='The number of filters'
                        )
    parser.add_argument('--filter_size', type=int, default=3,
                        help='The size of convolutional kernel'
                        )
    parser.add_argument('--up_type', choices=['interpolation', 'subpixel', 'transpose'],
                        default = 'transpose',
                        help='which type of upsampling operator is used'
                        )
    parser.add_argument('--repeat_num', type=int, default=1,
                        help='Repeated sampling numbers'
                        )
    parser.add_argument('--d_cnnx', type=int, default=8,
                        help='The depth of CNN_x'
                        )
    parser.add_argument('--d_cnnz', type=int, default=8,
                        help='The depth of CNN_z'
                        )
    parser.add_argument('--d_cnnm', type=int, default=8,
                        help='The depth of CNN_m'
                        )
    parser.add_argument('--setupUnet', type=bool, default=False,
                        help='If true, use Unet, else Resnet'
                        )
    parser.add_argument('--use_CA', type=bool, default=True,
                        help='If true, use channel attention'
                        )
    parser.add_argument('--BN', type=bool, default=False,
                        help='If true, use batch normalization'
                        )
    parser.add_argument('--bayesr_checkpoint', type=str, default = './models',
                        help = 'Path of BayeSR checkpoint '
                        )
    parser.add_argument('--downsamplor_checkpoint', type=str, default = './models',
                        help = 'Path of downsamplor checkpoint '
                        )
    parser.add_argument('--vgg_checkpoint', type=str, default = './models',
                        help = 'Path of VGG checkpoint '
                        )
    
    
def add_hyperpara_args(parser: ArgumentParser, FLAGS) -> None:
    parser.add_argument('--gammax', type=float, default=2.0,
                        help='Shape parameter of Gamma prior w.r.t. upsilon'
                        )
    parser.add_argument('--phix', type=float, default=1e-3,
                        help='Inverse scale parameter of Gamma prior w.r.t. upsilon'
                        )
    parser.add_argument('--gammaz', type=float, default=2.0,
                        help='Shape paramter of Gamma prior w.r.t. omega'
                        )
    parser.add_argument('--phiz', type=float, default=1e-3,
                        help='Inverse scale paramter of Gamma prior w.r.t. omega'
                        )
    gamman = {'DEN': 2.0, 'BiSR': 2.0, 'SySR': 8.0, 'ReSR': 2.0, 'RWSR': 2.0}[FLAGS.task]
    parser.add_argument('--gamman', type=float, default=gamman,
                        help='Shape paramter of Gamma prior w.r.t. rho'
                        )
    phin = {'supervised': 1e-3, 'pseudosupervised': 1e-5, 'unsupervised': 1e-5}[FLAGS.train_type]
    parser.add_argument('--phin', type=float, default=phin,
                        help='Inverse scale paramter of Gamma prior w.r.t. rho'
                        )
    parser.add_argument('--sigma0', type=float, default=1e6,
                        help='Inverse variance of noise mean'
                        )
    parser.add_argument('--tau', type=float, default=1.0,
                        help='Loss weight for discriminative learning'
                        )    
    lamda = {None: 0., 'GAN': 1e-4, 'WGAN': 1e-2, 'LSGAN': 1e-2}[FLAGS.GAN_type]
    parser.add_argument('--lamda', type=float, default=lamda,
                        help='Loss weight for generative adversarial learning'
                        )
