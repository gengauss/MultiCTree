#!/usr/bin/env python

# 前面ラベル数
numLabelA = 13
# 後面ラベル数
numLabelP = 12
# 後面ラベル数
numLabelP2 = 14
# 前後面ラベル数合計
numLabel = numLabelA + numLabelP

# 前面器官数
numStructureA = 12
# 後面器官数
numStructureP = 10

imageDirName = '../../../../anh/dataset/image'
labelDirName = '../../..//../anh/dataset/bone'
EDDirName = '../../..//../hara/from_datasets//ED'
atlasDirName = 'atlas_test'

imgHeight = 576
imgWidth = 256

StructureListA = ['skull', 'cervical_vertebrae', 'thoracie_vertebrae', 'lumbar_vertebrae', 'sacrum', 'pelvis', 'ribs', 'scapula', 'humerus', 'femur', 'sternum', 'clavicle']
StructureListP = ['skull', 'cervical_vertebrae', 'thoracie_vertebrae', 'lumbar_vertebrae', 'sacrum', 'pelvis', 'ribs', 'scapula', 'humerus', 'femur']

StructureListA2 = ['bkg_mask', 'skull', 'cervical_vertebrae', 'thoracie_vertebrae', 'lumbar_vertebrae', 'sacrum', 'pelvis', 'ribs', 'scapula', 'humerus', 'femur', 'sternum', 'clavicle']
StructureListP2 = ['bkg_mask', 'skull', 'cervical_vertebrae', 'thoracie_vertebrae', 'lumbar_vertebrae', 'sacrum', 'pelvis', 'ribs', 'scapula', 'humerus', 'femur', 'scapula2']

StructureListA3 = ['bkg_mask', 'skull', 'cervical_vertebrae', 'thoracie_vertebrae', 'lumbar_vertebrae', 'sacrum', 'pelvis', 'ribs', 'scapula', 'humerus', 'femur', 'sternum', 'clavicle']
StructureListP3 = ['bkg_mask', 'skull', 'cervical_vertebrae', 'thoracie_vertebrae', 'lumbar_vertebrae', 'sacrum', 'pelvis', 'ribs', 'scapula', 'humerus', 'femur', 'scapula2', 'scapula3', 'ribs3']
