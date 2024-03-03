import argparse
import logging
import os
import random
from re import A
import sys
from matplotlib.pyplot import sca
import numpy as np
from utils import test_single
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from TransBtrflyNet_dataset import TransBtrflyNet_dataset, RandomGenerator
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.vit_seg_modeling import TransBtrflyNet
from torchvision import transforms
import SimpleITK as sitk

parser = argparse.ArgumentParser()
# parser.add_argument('--gpu', '-g', type=int, default=-1,
#                         help='GPU ID (negative value indicates CPU)')
parser.add_argument('--root_path', type=str,
                    default='../dataset', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='TransBtrflyNet', help='experiment_name')
parser.add_argument('--numLabelA', type=int,
                    default=13, help='output channel of network A')
parser.add_argument('--numLabelP', type=int,
                    default=14, help='output channel of network P')

parser.add_argument('--list_dir', type=str,
                    default='./lists/1fold/test_list2.txt', help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=576, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.075, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
parser.add_argument('--fold', type=int, default=1, help='fold 1')
parser.add_argument('--gpu', '-g', type=str, default=-1,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_list2", list_dir=args.list_dir, transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    logging.info("{} test iterations per epoch".format(len(testloader)))

    # validation
    model.eval()
    metric_list = 0.0
    import pandas as pd
    classes_A = ['case_name', 'bkg_mask', 'skull', 'cervical_vertebrae', 'thoracic_vertebrae', 'lumbar_vertebrae', 'sacrum', 'pelvis', 'ribs', 'scapula', 'humerus', 'femur', 'sternum', 'clavicle']
    classes_P = ['case_name', 'bkg_mask', 'skull', 'cervical_vertebrae', 'thoracic_vertebrae', 'lumbar_vertebrae', 'sacrum', 'pelvis', 'ribs', 'scapula', 'humerus', 'femur', 'scapula2', 'scapula3', 'ribs2']
    caseA = []
    bkg_maskA = []
    skullA = []
    cervical_vertebraeA = []                                                 
    thoracic_vertebraeA = []
    lumbar_vertebraeA = []
    sacrumA = []
    pelvisA = []
    ribsA = []
    scapulaA = []
    humerusA = []
    femurA = []
    sternumA = [] 
    clavicleA = []

    caseP = []
    bkg_maskP = []
    skullP = []
    cervical_vertebraeP = []                                                 
    thoracic_vertebraeP = []
    lumbar_vertebraeP = []
    sacrumP = []
    pelvisP = []
    ribsP = []
    scapulaP = []
    humerusP = []
    femurP = []
    scapula2P = []
    scapula3P = []
    ribs2P = []

    dfA = pd.DataFrame(columns = classes_A)
    dfP = pd.DataFrame(columns = classes_P)
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label_batchA, label_batchP, case_name = sampled_batch['image'], sampled_batch['label'][0], sampled_batch['label'][1], sampled_batch['case_name'][i_batch][0]
        metric_i = test_single(image, label_batchA, label_batchP, model, args.numLabelA, args.numLabelP, 
                                       patch_size=[args.img_size, args.img_size],
                                       test_save_path=test_save_path, case=case_name)
        # metric_list += np.array(metric_i[0])
        # logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i[0][0], axis=0)[0], np.mean(metric_i[0][0], axis=0)[1]))
        # Anterior
        caseA.append(case_name)
        bkg_maskA.append(metric_i[0][0][0])
        skullA.append(metric_i[0][1][0])
        cervical_vertebraeA.append(metric_i[0][2][0])
        thoracic_vertebraeA.append(metric_i[0][3][0])
        lumbar_vertebraeA.append(metric_i[0][4][0])
        sacrumA.append(metric_i[0][5][0])
        pelvisA.append(metric_i[0][6][0])
        ribsA.append(metric_i[0][7][0])
        scapulaA.append(metric_i[0][8][0])
        humerusA.append(metric_i[0][9][0])
        femurA.append(metric_i[0][10][0])
        sternumA.append(metric_i[0][11][0])
        clavicleA.append(metric_i[0][12][0])

        # Posterior
        caseP.append(case_name)
        bkg_maskP.append(metric_i[1][0][0])
        skullP.append(metric_i[1][1][0])
        cervical_vertebraeP.append(metric_i[1][2][0])
        thoracic_vertebraeP.append(metric_i[1][3][0])
        lumbar_vertebraeP.append(metric_i[1][4][0])
        sacrumP.append(metric_i[1][5][0])
        pelvisP.append(metric_i[1][6][0])
        ribsP.append(metric_i[1][7][0])
        scapulaP.append(metric_i[1][8][0])
        humerusP.append(metric_i[1][9][0])
        femurP.append(metric_i[1][10][0])
        scapula2P.append(metric_i[1][11][0])
        scapula3P.append(metric_i[1][12][0])
        ribs2P.append(metric_i[1][13][0])

        # np.save('./t-SNE/input/1028/anterior/{}'.format(case_name), metric_i[4].squeeze(0).cpu())
        # np.save('./t-SNE/input/1028/posterior/{}'.format(case_name), metric_i[5].squeeze(0).cpu())

        # predict
        y1, y2 = metric_i[2], metric_i[3]

        # softmax
        softmax = nn.Softmax(dim=0)
        y1_soft = softmax(y1[0])
        y2_soft = softmax(y2[0])

        # argmax
        y1_argmax = y1_soft.max(0)[1]
        y2_argmax = y2_soft.max(0)[1]

        # 取り出してCPUに渡し，numpyに変換
        nda = y1_argmax.data
        outputA = nda.cpu().numpy()
        outputA = outputA.astype(np.uint8)

        ndp = y2_argmax.data
        outputP = ndp.cpu().numpy()
        outputP = outputP.astype(np.uint8)

        # Write mhd and png
        img_mhd_A = sitk.GetImageFromArray(outputA)
        img_mhd_A_ = sitk.GetImageFromArray(y1_soft.data.cpu().numpy())
        img_mhd_A.SetSpacing([2.8, 2.8])
        path_tmp = os.path.join('./results/anterior/test/{}fold_1/raw'.format(str(args.fold)), case_name + '_A.mhd')
        sitk.WriteImage(img_mhd_A, path_tmp, True)
        sitk.WriteImage(img_mhd_A_, os.path.join('./results/anterior/test/{}fold_1/pred'.format(str(args.fold)), case_name + '_A.mhd'))

        img_mhd_P = sitk.GetImageFromArray(outputP)
        img_mhd_P_ = sitk.GetImageFromArray(y2_soft.data.cpu().numpy())
        img_mhd_P.SetSpacing([2.8, 2.8])
        path_tmp = os.path.join('./results/posterior/test/{}fold_1/raw'.format(str(args.fold)), case_name + '_P.mhd')
        sitk.WriteImage(img_mhd_P, path_tmp, True)
        sitk.WriteImage(img_mhd_P_, os.path.join('./results/posterior/test/{}fold_1/pred'.format(str(args.fold)), case_name + '_P.mhd'))
    
    dfA['case_name'] = caseA
    dfA['bkg_mask'] = bkg_maskA
    dfA['skull'] = skullA
    dfA['cervical_vertebrae'] = cervical_vertebraeA
    dfA['thoracic_vertebrae'] = thoracic_vertebraeA
    dfA['lumbar_vertebrae'] = lumbar_vertebraeA
    dfA['sacrum'] = sacrumA
    dfA['pelvis'] = pelvisA
    dfA['ribs'] = ribsA
    dfA['scapula'] = scapulaA
    dfA['humerus'] = humerusA
    dfA['femur'] = femurA
    dfA['sternum'] = sternumA
    dfA['clavicle'] = clavicleA

    dfP['case_name'] = caseP
    dfP['bkg_mask'] = bkg_maskP
    dfP['skull'] = skullP
    dfP['cervical_vertebrae'] = cervical_vertebraeP
    dfP['thoracic_vertebrae'] = thoracic_vertebraeP
    dfP['lumbar_vertebrae'] = lumbar_vertebraeP
    dfP['sacrum'] = sacrumP
    dfP['pelvis'] = pelvisP
    dfP['ribs'] = ribsP 
    dfP['scapula'] = scapulaP
    dfP['humerus'] = humerusP
    dfP['femur'] = femurP
    dfP['scapula2'] = scapula2P
    dfP['scapula3'] = scapula3P
    dfP['ribs2'] = ribs2P

    dfA.to_excel('./results/anterior/test/{}fold_1/all-A_DICE.xlsx'.format(str(args.fold)))
    dfP.to_excel('./results/posterior/test/{}fold_1/all-P_DICE.xlsx'.format(str(args.fold)))
    return "Testing Finished!"


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

    dataset_config = {
        'TransBtrflyNet': {
            'Dataset': TransBtrflyNet_dataset,
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'numLabelA': args.numLabelA,
            'numLabelP': args.numLabelP,
        },
    }

    dataset_name = args.dataset
    args.numLabelA = dataset_config[dataset_name]['numLabelA']
    args.numLabelP = dataset_config[dataset_name]['numLabelP']
    args.volume_path = dataset_config[dataset_name]['root_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True


    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = TransBtrflyNet(config_vit, img_size=args.img_size).cuda()

    # Load best model
    net.load_state_dict(torch.load('./models/{}fold/finetune/epoch_99.pth'.format(args.fold)))
    test_save_path = None
    inference(args, net, test_save_path)
