import argparse
import logging
import os
import random
from re import I
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLossDSV, multiCTree
from torchvision import transforms
import higra as hg
import torch.nn.functional as F
import pandas as pd
import csv

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

def trainer_bonescinti(args, model, snapshot_path):
    from TransBtrflyNet_dataset import TransBtrflyNet_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    numLabelA = args.numLabelA
    numLabelP = args.numLabelP
    batch_size = args.batch_size * args.n_gpu
    max_iterations = args.max_iterations
    db_train = TransBtrflyNet_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
    transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()

    dice_lossA = DiceLossDSV(numLabelA)
    dice_lossP = DiceLossDSV(numLabelP)
    dice_lossA_CT = multiCTree(numLabelA)
    dice_lossP_CT = multiCTree(numLabelP)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    la = []
    lca = []
    lda = []
    lcta = []
    lp = []
    lcp = []
    ldp = []
    lctp = []
    l = []

    # Initialize CSV file and writer
    csv_filenameA = os.path.join(snapshot_path, 'log_A.csv')
    csv_fileA = open(csv_filenameA, mode='w', newline='')
    csv_writerA = csv.writer(csv_fileA)
    csv_writerA.writerow(['iteration', 'loss', 'loss_CE', 'loss_dice', 'loss_CT'])

    csv_filenameP = os.path.join(snapshot_path, 'log_P.csv')
    csv_fileP = open(csv_filenameP, mode='w', newline='')
    csv_writerP = csv.writer(csv_fileP)
    csv_writerP.writerow(['iteration', 'loss', 'loss_CE', 'loss_dice', 'loss_CT'])

    csv_filename= os.path.join(snapshot_path, 'log.csv')
    csv_file = open(csv_filename, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['iteration', 'loss'])

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image, label_batchA, label_batchP = sampled_batch['image'], sampled_batch['label'][0], sampled_batch['label'][1]
            image, label_batchA, label_batchP = image.cuda(), label_batchA.cuda(), label_batchP.cuda()

            outputsA = model(image)[0]
            outputsP = model(image)[1]

            outputsAM = model(image)[2]
            outputsPM = model(image)[3]

            outputsAL = model(image)[4]
            outputsPL = model(image)[5]

            loss_ceA = ce_loss(outputsA, label_batchA.long())
            loss_ceP = ce_loss(outputsP, label_batchP.long())

            loss_diceA = dice_lossA(outputsA, outputsAM, outputsAL, label_batchA, softmax=True)
            loss_diceP = dice_lossP(outputsP, outputsPM, outputsPL, label_batchP, softmax=True)

            optimizer.zero_grad()
            
            loss_ct_A = dice_lossA_CT(outputsA, label_batchA, softmax=True)
            loss_ct_P = dice_lossP_CT(outputsP, label_batchP, softmax=True)

            lossA = 0.5 * loss_ceA + 0.5 * loss_diceA + 0.001 * loss_ct_A
            lossP = 0.5 * loss_ceP + 0.5 * loss_diceP + 0.001 * loss_ct_P
            loss = 0.5 * (lossA + lossP)

            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            logging.info('Iteration %d : loss : %f loss_dice : %f loss_CT : %f' % (iter_num, loss.item(), (loss_diceA.item() + loss_diceP.item())/2, (loss_ct_A.item() + loss_ct_P.item())/2))
            la.append(lossA.item())
            lca.append(loss_ceA.item())
            lda.append(loss_diceA.item())
            lcta.append(loss_ct_A.item())

            lp.append(lossP.item())
            lcp.append(loss_ceP.item())
            ldp.append(loss_diceP.item())
            lctp.append(loss_ct_P.item())

            l.append(loss.item())

            iteration = epoch_num * len(trainloader) + i_batch
            csv_writerA.writerow([iteration, lossA.item(), loss_ceA.item(), loss_diceA.item(), loss_ct_A.item()])
            csv_writerP.writerow([iteration, lossP.item(), loss_ceP.item(), loss_diceP.item(), loss_ct_P.item()])
            csv_writer.writerow([iteration, loss.item()])
            

        save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num+88) + '.pth')
        torch.save(model.state_dict(), save_mode_path)
        logging.info("save model to {}".format(save_mode_path))

    writer.close()
    csv_file.close()
    return "Training Finished!"