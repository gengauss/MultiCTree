from fileinput import filename
import os

from traitlets import Type
import config
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import SimpleITK as sitk


# Random rotation flip
def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


# Random rotation
def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def flip(x):
    if len(x.shape) == 2:
        x[:,::-1].copy()
    elif len(x.shape) == 3:
        x[:,:,::-1].copy()
    else:
        "dimension error"

    return x

def read_mhd_and_raw(path, numpyFlag=True):
    img = sitk.ReadImage(path)
    if not numpyFlag:
        return img  
    return img

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label0, label1, label0s, label1s = sample['image'], sample['label'][0], sample['label'][1], sample['label'][2], sample['label'][3]
        if np.random.rand() > 0.5:
            image = flip(image)
            label0 = flip(label0)
            label1 = flip(label1)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(1)
        label0 = torch.from_numpy(label0.astype(np.float32)).long()
        label1 = torch.from_numpy(label1.astype(np.float32)).long()
        label0s = torch.from_numpy(label0s.astype(np.float32)).long()
        label1s = torch.from_numpy(label1s.astype(np.float32)).long()
        label = (label0, label1, label0s, label1s)
        sample = {'image': image, 'label': label}
        return sample


class TransBtrflyNet_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        print('Initalize dataset')
        self.transform = transform  # using transform in torch!
        self.split = split
        self.data_dir = base_dir
        self.list_dir = list_dir

        # Input name list
        with open(self.list_dir) as f:
            all_line = f.readlines()
            self.file_name = [line.replace("\n","") for line in all_line]

        self.case_list = []
        for i in range(len(self.file_name)):
            each_data = []
            # image
            each_data.append(os.path.join(self.data_dir, 'image', self.file_name[i]+'_A.mhd')) #x1
            each_data.append(os.path.join(self.data_dir, 'image', self.file_name[i]+'_P.mhd')) #x2

            #label
            each_data.append(os.path.join(self.data_dir, 'bone', self.file_name[i]+'_A.mhd')) #t1
            each_data.append(os.path.join(self.data_dir, 'bone', self.file_name[i]+'_P.mhd')) #t2

            #

            self.case_list.append(each_data)
        
        print('Initalize done')

    def flip(self, x):
        if len(x.shape) == 2:
            x[:,::-1].copy()
        elif len(x.shape) == 3:
            x[:,:,::-1].copy()
        else:
            "dimension error"

    def __len__(self):
        return len(self.file_name)

    def __getitem__(self, idx):
        if self.split == "train":
            # slice_name = self.sample_list[idx].strip('\n')
            
            # input image
            x = np.concatenate((sitk.GetArrayFromImage(read_mhd_and_raw(self.case_list[idx][0], np.float32))[np.newaxis,:,:], 
                                sitk.GetArrayFromImage(read_mhd_and_raw(self.case_list[idx][1], np.float32))[np.newaxis,:,:]), axis = 0)
            # ans label
            t1 = sitk.GetArrayFromImage(read_mhd_and_raw(self.case_list[idx][2], np.float32))
            t2 = sitk.GetArrayFromImage(read_mhd_and_raw(self.case_list[idx][3], np.float32))

            # sepalate label
            t1s = np.identity(config.numLabelA, dtype = np.bool)[t1].transpose(2, 0, 1).astype(np.float32)
            t2s = np.identity(config.numLabelP, dtype = np.bool)[t2].transpose(2, 0, 1).astype(np.float32)

            image  = x
            label = (t1, t2, t1s, t2s)

        else:
            # input image
            x = np.concatenate((sitk.GetArrayFromImage(read_mhd_and_raw(self.case_list[idx][0], np.float32))[np.newaxis,:,:], 
                                sitk.GetArrayFromImage(read_mhd_and_raw(self.case_list[idx][1], np.float32))[np.newaxis,:,:]), axis = 0)
            # ans label
            t1 = sitk.GetArrayFromImage(read_mhd_and_raw(self.case_list[idx][2], np.float32))
            t2 = sitk.GetArrayFromImage(read_mhd_and_raw(self.case_list[idx][3], np.float32))

            # sepalate label
            t1s = np.identity(config.numLabelA, dtype = np.bool)[t1].transpose(2, 0, 1).astype(np.float32)
            t2s = np.identity(config.numLabelP, dtype = np.bool)[t2].transpose(2, 0, 1).astype(np.float32)

            image  = x
            label = (t1, t2, t1s, t2s)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.file_name
        return sample
