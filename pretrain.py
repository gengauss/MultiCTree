import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import TransBtrflyNet
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from pretrainer import pretrainer_bonescinti

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../dataset', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='TransBtrflyNet', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/1fold/train.txt', help='list dir')
# Output channel of network
parser.add_argument('--numLabelA', type=int,
                    default=13, help='output channel of network A')
parser.add_argument('--numLabelP', type=int,
                    default=14, help='output channel of network P')

parser.add_argument('--max_iterations', type=int,
                    default=3, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=5, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate') # Best 0.1, CT: 0.001
parser.add_argument('--img_size', type=int,
                    default=576, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=8, help='vit_patches_size, default is 16')
parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
parser.add_argument('--fold', type=int, default=1, help='fold 1')
parser.add_argument('--loss', '-l', default="DiceDsv", help='Loss function for update')
parser.add_argument('--gpu', '-g', type=str, default=-1,
                        help='GPU ID (negative value indicates CPU)')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'TransBtrflyNet': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'numLabelA': args.numLabelA,
            'numLabelP': args.numLabelP,
        },
    }

    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.numLabelA = dataset_config[dataset_name]['numLabelA']
    args.numLabelP = dataset_config[dataset_name]['numLabelP']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    snapshot_path = "./models/{}fold/pretrain/".format(args.fold)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = TransBtrflyNet(config_vit, img_size=args.img_size).cuda()
    CDir = os.path.dirname(os.path.abspath(__file__))
    
    if args.resume:
        # Resume from a snapshot
        initial_model_path = os.path.join(CDir, 'models', args.resume)
        net.load_state_dict(torch.load(initial_model_path))

    trainer = {'TransBtrflyNet': pretrainer_bonescinti,}
    trainer[dataset_name](args, net, snapshot_path)
