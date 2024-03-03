import numpy as np
import torch as tc
import higra as hg
import imageio

import math
from torch.nn import Module
from torch.autograd import Function
import matplotlib.pyplot as plt
import torch.nn.functional as F

try:
    from utils import * # imshow, locate_resource
except: # we are probably running from the cloud, try to fetch utils functions from URL
    import urllib.request as request; exec(request.urlopen('https://github.com/higra/Higra-Notebooks/raw/master/utils.py').read(), globals())


class ComponentTreeFunction(Function):
  @staticmethod
  def forward(ctx, graph, vertex_weights, tree_type="max", plateau_derivative="full"):
    """
    Construct a component tree of the given vertex weighted graph.

    tree_type must be in ("min", "max", "tos")

    plateau_derivative can be "full" or "single". In the first case, the gradient of an altitude component
    is back-propagated to the vertex weights of the whole plateau (to all proper vertices of the component).
    In the second case, an arbitrary vertex of the plateau is selected and will receive the gradient.

    return: the altitudes of the tree (torch tensor), the tree itself is stored as an attribute of the tensor
    """
    if tree_type == "max":
      tree, altitudes = hg.component_tree_max_tree(graph, vertex_weights.detach().numpy())
    elif tree_type == "min":
      tree, altitudes = hg.component_tree_min_tree(graph, vertex_weights.detach().numpy())
    elif tree_type == "tos":
      tree, altitudes = hg.component_tree_tree_of_shapes_image2d(vertex_weights.detach().numpy())
    else:
      raise ValueError("Unknown tree type " + str(tree_type))

    if plateau_derivative == "full":
      plateau_derivative = True
    elif plateau_derivative == "single":
      plateau_derivative = False
    else:
      raise ValueError("Unknown plateau derivative type " + str(plateau_derivative))
    ctx.saved = (tree, graph, plateau_derivative)
    altitudes = tc.from_numpy(altitudes).clone().requires_grad_(True)
    # torch function can only return tensors, so we hide the tree as a an attribute of altitudes
    altitudes.tree = tree
    return altitudes

  @staticmethod
  def backward(ctx, grad_output):
    tree, graph, plateau_derivative = ctx.saved
    if plateau_derivative:
      grad_in = grad_output[tree.parents()[:tree.num_leaves()]]
    else:
      leaf_parents = tree.parents()[:tree.num_leaves()]
      _, indices = np.unique(leaf_parents, return_index=True)
      grad_in = tc.zeros((tree.num_leaves(),), dtype=grad_output.dtype)
      grad_in[indices] = grad_output[leaf_parents[indices]]
    return None, hg.delinearize_vertex_weights(grad_in, graph), None

class ComponentTree(Module):
    def __init__(self, tree_type):
        super().__init__()
        tree_types = ("max", "min", "tos")
        if tree_type not in tree_types:
          raise ValueError("Unknown tree type " + str(tree_type) + " possible values are " + " ".join(tree_types))

        self.tree_type = tree_type

    def forward(self, graph, vertex_weights):
        altitudes = ComponentTreeFunction.apply(graph, vertex_weights, self.tree_type)
        return altitudes.tree, altitudes

max_tree = ComponentTree("max")
min_tree = ComponentTree("min")
tos_tree = ComponentTree("tos")


class Optimizer:
    def __init__(self, loss, lr, optimizer="adam"):
        """
        Create an Optimizer utility object

        loss: function that takes a single torch tensor which support requires_grad = True and returns a torch scalar
        lr: learning rate
        optimizer: "adam" or "sgd"
        """
        self.loss_function = loss
        self.history = []
        self.optimizer = optimizer
        self.lr = lr
        self.best = None
        self.best_loss = float("inf")

    def fit(self, data, iter=1000, debug=False, min_lr=1e-6):
        """
        Fit the given data

        data: torch tensor, input data
        iter: int, maximum number of iterations
        debug: int, if > 0, print current loss value and learning rate every debug iterations
        min_lr: float, minimum learning rate (an LR scheduler is used), if None, no LR scheduler is used 
        """
        data = data.clone().requires_grad_(True)
        if self.optimizer == "adam":
            optimizer = tc.optim.Adam([data], lr=self.lr, amsgrad=True)
        else:
            optimizer = tc.optim.SGD([data], lr=self.lr)

        if min_lr:
            lr_scheduler = tc.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=100)
    
        for t in range(iter):
            optimizer.zero_grad()
            kernel = torch.ones(1, 1, 5, 5).to(data.device)
            eroded_data = F.conv2d(data.unsqueeze(0).unsqueeze(0).to(torch.double), kernel.to(torch.double), padding=(kernel.size(2) - 1) // 2).squeeze().float()
            
            loss = self.loss_function(tc.relu(eroded_data))

            loss.backward()
            optimizer.step()  
            loss_value = loss.item()
            
            self.history.append(loss_value) 
            if loss_value < self.best_loss:
                self.best_loss = loss_value
                self.best = tc.relu(data).clone()
                
            if min_lr:
                lr_scheduler.step(loss_value)
                if optimizer.param_groups[0]['lr'] <= min_lr:
                    break

            if debug and t % debug == 0:
                print("Iteration {}: Loss: {:.4f}, LR: {}".format(t, loss_value, optimizer.param_groups[0]['lr']))
        return self.best

    def show_history(self):
        """
        Plot loss history
        """
        plt.plot(self.history)
        plt.show()


def attribute_depth(tree, altitudes):
    """
    Compute the depth of any node of the tree which is equal to the largest altitude 
    in the subtree rooted in the current node. 

    :param tree: input tree
    :param altitudes: np array (1d), altitudes of the input tree nodes
    :return: np array (1d), depth of the tree nodes
    """
    return hg.accumulate_sequential(tree, altitudes[:tree.num_leaves()], hg.Accumulators.max)

def attribute_saddle_nodes(tree, attribute):
    """
    Let n be a node and let an be an ancestor of n. The node an has a single child node that contains n denoted by ch(an -> n). 
    The saddle and base nodes associated to a node n for the given attribute values are respectively the closest ancestor an  
    of n and the node ch(an -> n) such that there exists a child c of an with attr(ch(an -> n)) < attr(c). 

    :param tree: input tree
    :param attribute: np array (1d), attribute of the input tree nodes
    :return: (np array, np array), saddle and base nodes of the input tree nodes for the given attribute
    """
    max_child_index = hg.accumulate_parallel(tree, attribute, hg.Accumulators.argmax)
    child_index = hg.attribute_child_number(tree)
    main_branch = child_index == max_child_index[tree.parents()]
    main_branch[:tree.num_leaves()] = True

    saddle_nodes = hg.propagate_sequential(tree, np.arange(tree.num_vertices())[tree.parents()], main_branch)
    base_nodes = hg.propagate_sequential(tree, np.arange(tree.num_vertices()), main_branch)
    return saddle_nodes, base_nodes

def attribute_dice(image_gt, image, tree):
    nbGTPix = image_gt.sum()
    areaNodes = hg.attribute_area(tree)
    gt = np.reshape(image_gt,len(image_gt))
    image = np.reshape(image,len(image))
    att = hg.accumulate_sequential(tree, (image != 0) & (gt != 0), hg.Accumulators.sum)
    union = nbGTPix + areaNodes
    dice = 2 * att / union
    return hg.accumulate_and_max_sequential(tree, dice, dice[:len(image_gt)], hg.Accumulators.max)


def loss_maxima(graph, image, image_gt, saliency_measure, importance_measure, margin=1):
    tree, altitudes = max_tree(graph, image)
    altitudes_np = altitudes.detach().numpy()

    extrema = hg.attribute_extrema(tree, altitudes_np)
    extrema_indices = np.arange(tree.num_vertices())[extrema]
    extrema_altitudes = altitudes[tc.from_numpy(extrema_indices).long()]

    if importance_measure == "area":
        area = hg.attribute_area(tree)
        pass_nodes, base_nodes = attribute_saddle_nodes(tree, area)
        extrema_area = tc.from_numpy(area[base_nodes[extrema_indices]])

    if importance_measure == "volume":
        volume = hg.attribute_volume(tree, altitudes_np)
        pass_nodes, base_nodes = attribute_saddle_nodes(tree, volume)
        extrema_volume = tc.from_numpy(volume[base_nodes[extrema_indices]])

    saliency = []
    if saliency_measure[0] == "altitude" and saliency_measure[1] == "connect":
        depth = attribute_depth(tree, altitudes_np)
        saddle_nodes = tc.from_numpy(attribute_saddle_nodes(tree, depth)[0]).long()
        extrema_connect = 1 - altitudes[saddle_nodes[extrema_indices]]
        saliency.append(extrema_altitudes)
        saliency.append(extrema_connect)

    if importance_measure == "altitude":
        importance = extrema_altitudes
    elif importance_measure == "dynamics":
        depth = attribute_depth(tree, altitudes_np)
        saddle_nodes = tc.from_numpy(attribute_saddle_nodes(tree, depth)[0])
        extrema_dynamics = extrema_altitudes - altitudes[saddle_nodes[extrema_indices].long()]
        importance = extrema_dynamics

    elif importance_measure == "area":
        importance = extrema_area
    elif importance_measure == "volume":
        importance = extrema_volume

    elif importance_measure == "dice":
        image_gt = image_gt.reshape(576*256, 1)
        image = image.detach().numpy().reshape(576*256, 1)
        dice = attribute_dice(image_gt, image, tree)
        extinction_value = hg.attribute_extinction_value(tree, altitudes_np, np.array(dice))
        importance = [tc.tensor(extinction_value[i]) for i in extrema_indices]

    G1 = []
    G2 = []
    G3 = []

    for i, ext_idx in enumerate(extrema_indices):
        if importance[i] == 0:
            G1.append(i)
        elif importance[i] >= 0.7:
            G2.append(i)
        else:
            G3.append(i)

    return tc.sum(saliency[0][G1]**p) + tc.sum(tc.relu(margin - saliency[0][G2])**p) + tc.sum(saliency[1][G3]**p)