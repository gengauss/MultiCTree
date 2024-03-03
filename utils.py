import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from scipy import ndimage
from multiCTree import loss_maxima
import higra as hg
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        print(self.n_classes)

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss
    
    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class orgCTree(nn.Module):
    def __init__(self, n_classes):
        super(orgCTree, self).__init__()
        self.n_classes = n_classes
        print(self.n_classes)

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    
    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        loss = 0
        # anterior
        if self.n_classes == 13:
            for i in range(0, self.n_classes):
                graph = hg.get_8_adjacency_implicit_graph(inputs[:,i][0,:,:].shape)
                if i == 0 or i == 1 or i == 2 or i == 4 or i == 5 or i == 6 or i == 11:
                    loss_ct = loss_maxima(graph, inputs[:,i][0,:,:].cpu().data, "dynamics", "volume", num_target_maxima=1, margin=1, p=1)
                if i == 3 or i == 9 or i == 10 or i == 12:
                    loss_ct = loss_maxima(graph, inputs[:,i][0,:,:].cpu().data, "dynamics", "volume", num_target_maxima=2, margin=1, p=1)
                if i == 7 or i == 8:
                    loss_ct = loss_maxima(graph, inputs[:,i][0,:,:].cpu().data, "dynamics", "volume", num_target_maxima=4, margin=1, p=1)
                    
                loss += loss_ct
                
        # posterior
        elif self.n_classes == 14:
            for i in range(0, self.n_classes):
                graph = hg.get_8_adjacency_implicit_graph(inputs[:,i][0,:,:].shape)
                if i == 0 or i == 1 or i == 2 or i == 3 or i == 4 or i == 5 or i == 6:
                    loss_ct = loss_maxima(graph, inputs[:,i][0,:,:].cpu().data, "dynamics", "volume", num_target_maxima=1, margin=1, p=1)
                if i == 7 or i == 8 or i == 9 or i == 10 or i == 11 or i == 12 or i == 13:
                    loss_ct = loss_maxima(graph, inputs[:,i][0,:,:].cpu().data, "dynamics", "volume", num_target_maxima=2, margin=1, p=1)
                
                loss += loss_ct
        return loss / self.n_classes   
    

class multiCTree(nn.Module):
    def __init__(self, n_classes):
        super(multiCTree, self).__init__()
        self.n_classes = n_classes
        print(self.n_classes)

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = 0
        for i in range(0, self.n_classes):
            graph = hg.get_8_adjacency_implicit_graph(inputs[:,i][0,:,:].shape)
            loss_ct = loss_maxima(graph, inputs[:,i][0,:,:].cpu().data, target[:,i][0,:,:].cpu().data.numpy(), ["altitude", "connect"], "dice", margin=1)
                
            loss += loss_ct

        return loss / self.n_classes 
    

class DiceLossDSV(nn.Module):
    def __init__(self, n_classes):
        super(DiceLossDSV, self).__init__()
        self.n_classes = n_classes
        print(self.n_classes)

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, inputsM, inputsL, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
            inputsM = torch.softmax(inputsM, dim=1)
            inputsL = torch.softmax(inputsL, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            diceU = self._dice_loss(inputs[:, i], target[:, i])
            diceM = self._dice_loss(inputsM[:, i], target[:, i])
            diceL = self._dice_loss(inputsL[:, i], target[:, i])
            dice = (diceU + diceM + diceL) / 3
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]

        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single(image, labelA, labelP, net, numLabelA, numLabelP, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, labelA, labelP = image.squeeze(0).cpu().detach().numpy(), labelA.squeeze(0).cpu().detach().numpy(), labelP.squeeze(0).cpu().detach().numpy()
    input = torch.from_numpy(image).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        outputs = net(input)
        outA = torch.argmax(torch.softmax(outputs[0], dim=1), dim=1).squeeze(0)
        outP = torch.argmax(torch.softmax(outputs[1], dim=1), dim=1).squeeze(0)
        predictionA = outA.cpu().detach().numpy()
        predictionP = outP.cpu().detach().numpy()

    metric_listA = []
    metric_listP = []
    classes_A = ['case_name', 'bkg_mask', 'skull', 'cervical_vertebrae', 'thoracie_vertebrae', 'lumbar_vertebrae', 'sacrum', 'pelvis', 'ribs', 'scapula', 'humerus', 'femur', 'sternum', 'clavicle']
    classes_P = ['case_name', 'bkg_mask', 'skull', 'cervical_vertebrae', 'thoracie_vertebrae', 'lumbar_vertebrae', 'sacrum', 'pelvis', 'ribs', 'scapula', 'humerus', 'femur', 'scapula2', 'scapula3', 'ribs2']

    for i in range(0, numLabelA):
        metric_listA.append(calculate_metric_percase(predictionA == i, labelA == i))
        
    for i in range(0, numLabelP):
        if i < 12:
            metric_listP.append(calculate_metric_percase(predictionP == i, labelP == i))
        elif i == 12:
            metric_listP.append(calculate_metric_percase((predictionP == 8) | (predictionP == 11), (labelP == 8) | (labelP == 11)))
        else:
            metric_listP.append(calculate_metric_percase((predictionP == 7) | (predictionP == 11), (labelP == 7) | (labelP == 11)))
    return metric_listA, metric_listP, outputs[0], outputs[1], outputs[0].squeeze(0).cpu(), outputs[1].squeeze(0).cpu()
