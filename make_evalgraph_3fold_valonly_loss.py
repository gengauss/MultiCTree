#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import csv
import pandas as pd
import argparse
import math

np.set_printoptions(precision=3)

def chunked(iterable, n):
    return [iterable[x:x + n] for x in range(0, len(iterable), n)]

def main():
    args = sys.argv
    parser = argparse.ArgumentParser(
        description='chainer line drawing colorization')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--root', '-R', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Root directory path of image files')
    parser.add_argument('--inputA', '-inA', default=os.path.dirname(os.path.abspath(__file__)) + "/result/test/Group4",
                        help='Directory of image files.')
    parser.add_argument('--inputP', '-inP', default=os.path.dirname(os.path.abspath(__file__)) + "/result/test/Group4",
                        help='Directory of image files.')
    parser.add_argument('--output', '-o', default=os.path.dirname(os.path.abspath(__file__)) + "/result/EBGAN_SCE9_org/test/Group6",
                        help='Directory of image files.')
    parser.add_argument('--group', '-n', type=int, default=1,
                        help='Number of Group')
    parser.add_argument('--start', '-s', type=int, default=1,
                        help='for for')
    parser.add_argument('--end', '-e', type=int, default=100,
                        help='for for')
    parser.add_argument('--interval', '-i', type=int, default=1,
                        help='for for')
    parser.add_argument('--train', '-t', type=int, default=0,
                        help='train')
    args = parser.parse_args()

    inputA=args.inputA
    inputP=args.inputP
    output=args.output

    itr_list = [args.start,args.end,args.interval]
    itr_num = (itr_list[1] - itr_list[0]) / itr_list[2] + 1

    fold_set = ["Group1"]
    fold_n = len(fold_set)
    if args.train == 1:
        phase = ["validation", "training"]
    else:
        phase = ["validation"]
    phase_n = len(phase)
    if args.train == 1:
        phase2 = ['val_Loss-AP', 'train_Loss-AP']
    else:
        phase2 = ['val_Loss-AP']

    structure_Aname = ["total_loss"]
    structure_Pname = ["total_loss"]

    train_aveA = np.zeros((fold_n, phase_n, int(itr_num)), dtype = np.float32)
    train_aveP = np.zeros((fold_n, phase_n, int(itr_num)), dtype = np.float32)
    train_aveAP = np.zeros((fold_n, phase_n, int(itr_num)), dtype = np.float32)
    Loss_aveA = np.zeros((fold_n, phase_n, int(itr_num)), dtype = np.float32)
    Loss_aveP = np.zeros((fold_n, phase_n, int(itr_num)), dtype = np.float32)
    Loss_aveAP = np.zeros((fold_n, phase_n, int(itr_num)), dtype = np.float32)
    itr_list_ = np.zeros(int(itr_num), dtype = np.int32)
    
    for set_num in range (fold_n):
        for ph_num in range (phase_n):
            path_valA = inputA
            path_valP = inputP
            n=0
            for i in range (itr_list[0], itr_list[1]+1, itr_list[2]):            
                if ph_num == 0:
                    df_val_AD = pd.read_excel(path_valA + "/all-A_loss_epoch_{}.xlsx".format(str(i)))
                    df_val_PD = pd.read_excel(path_valP + "/all-P_loss_epoch_{}.xlsx".format(str(i)))
                    ave_AD = 0
                    ave_PD = 0
                    for Aname in structure_Aname:
                        ave_AD += df_val_AD[Aname].mean()
                    Loss_aveA[set_num, ph_num, n] = ave_AD/len(structure_Aname)
                    for Pname in structure_Pname:
                        ave_PD += df_val_PD[Pname].mean()
                    Loss_aveP[set_num, ph_num, n] = ave_PD/len(structure_Pname)

                    Loss_aveAP[set_num, ph_num, n] = 0.5 * (Loss_aveA[set_num, ph_num, n] + Loss_aveP[set_num, ph_num, n])

                elif ph_num == 1:
                    df_train_AD = pd.read_excel(path_valA + "/all-A_loss_epoch_{}.xlsx".format(str(i)))
                    df_train_PD = pd.read_excel(path_valP + "/all-P_loss_epoch_{}.xlsx".format(str(i)))
                    ave_AD = 0
                    ave_PD = 0
                    for Aname in structure_Aname:
                        ave_AD += df_train_AD[Aname].mean()
                    Loss_aveA[set_num, ph_num, n] = ave_AD/len(structure_Aname)
                    for Pname in structure_Pname:
                        ave_PD += df_train_PD[Pname].mean()
                    Loss_aveP[set_num, ph_num, n] = ave_PD/len(structure_Pname)

                    Loss_aveAP[set_num, ph_num, n] = 0.5 * (Loss_aveA[set_num, ph_num, n] + Loss_aveP[set_num, ph_num, n])

                itr_list_[n] = i

                n += 1

    # aveve => A,B,C's ave
    Loss_aveveA = np.zeros((phase_n, int(itr_num)), dtype = np.float32)
    Loss_aveveP = np.zeros((phase_n, int(itr_num)), dtype = np.float32)
    Loss_aveveAP = np.zeros((phase_n, int(itr_num)), dtype = np.float32)

    # ...aveve is average of A ,B and C
    Loss_aveveA = np.average(Loss_aveA, axis = 0)
    Loss_aveveP = np.average(Loss_aveP, axis = 0)
    Loss_aveveAP = np.average(Loss_aveAP, axis = 0)

    color_palette = ['red', 'green', 'orange', 'blue', 'magenta', 'lawngreen']

    for ph_num in range (phase_n):
        plt.plot(itr_list_, Loss_aveveAP[ph_num], '-x', label=phase2[ph_num], lw=2, color=color_palette[ph_num+1])
        if np.amin(Loss_aveveAP[ph_num]) == 1.:
            pass
        elif ph_num == 0:
            plt.plot(itr_list_[np.where(Loss_aveveAP[ph_num] == np.amin(Loss_aveveAP[ph_num]))], np.amin(Loss_aveveAP[ph_num]), 'o', markersize=10, color=color_palette[0])
            plt.text(itr_list_[np.where(Loss_aveveAP[ph_num] == np.amin(Loss_aveveAP[ph_num]))], np.amin(Loss_aveveAP[ph_num]), str(int(itr_list_[np.where(Loss_aveveAP[ph_num] == np.amin(Loss_aveveAP[ph_num]))])), ha='center', va='top', fontsize=10)

    print("Group{}_epoch = {}".format(args.group, itr_list_[np.where(Loss_aveveAP[0] == np.amin(Loss_aveveAP[0]))]))
    plt.grid(which="both")
    plt.ylim([0, 5])
    plt.legend(loc='upper right')
    plt.xlabel("epoch")
    plt.ylabel("Total loss")
    plt.tight_layout()
    # plt.show()
    plt.savefig("{}/allepoch_{}.png".format(output, args.group))
    plt.close()

if __name__ == '__main__':
    main()

