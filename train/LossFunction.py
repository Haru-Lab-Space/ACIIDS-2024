import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Union

class Focalloss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        '''
        :param inputs: batch_size * dim
        :param targets: (batch,)
        :return:
        '''
        bce_loss = F.cross_entropy(inputs, targets)
        loss = self.alpha * (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss
        return loss


# Binary implementation based on https://amaarora.github.io/2020/06/29/FocalLoss.html
# See this discussion https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/162035#904086
# Multiclass and multilabel generalization based on that code.

def focal_loss(input, target, fl_type: str, alphas, gamma, reduction):
    if fl_type not in ['binary', 'multiclass', 'multilabel']:
        raise ValueError(f"fl_type should be binary, multiclass or multilabel instead of {fl_type}.")
        
    # Checks (mainly copied from kornia.losses.focal_loss)
    ndims, shape_msg = (1, 'B (batch_size)') if fl_type == 'binary' else (2, 'BxC') 
    if input.ndim != ndims:
        raise ValueError(f"Invalid input shape, we expect {shape_msg}. Got: {input.shape}.")
        
    ndims, shape_msg = (2, 'BxC') if fl_type == 'multilabel' else (1, 'B (batch_size)') 
    if target.ndim != ndims:
        raise ValueError(f"Invalid target shape, we expect {shape_msg}. Got: {target.shape}.")
    
    if fl_type == 'multiclass':
        if input.shape[0] != target.shape[0]:
            raise ValueError(f'Expected input batch_size ({input.shape[0]}) to match target batch_size ({target.shape[0]}).')
            
        if target.max() >= input.shape[1]:
            raise ValueError(f"There are more target classes ({target.max()+1}) than the number of classes predicted ({input.shape[1]})")            
    else:
        if input.shape != target.shape:
            raise ValueError(f"Expected input shape ({input.shape}) to match target shape ({target.shape}).")      
        
    
    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}" .format(
                input.device, target.device))
        
    if gamma < 1:
        raise RuntimeError('Backpropagation problems. See EfficientDet Rwightman focal loss implementation')
        
    # Create at check alpha values
    # Create an alphas tensor. Remember to move it to the same device and to have the same dtype that the inputs
    if fl_type == 'binary' and (not isinstance(alphas, torch.Tensor) or alpha.ndim == 0): 
        if not 0 < alphas < 1: 
            raise ValueError(f"Alpha must be between 0 and 1 and it's {alphas}.")
        alphas = torch.tensor([alphas, 1-alphas], dtype=input.dtype, device=input.device) # [0, 1] labels weights
    elif isinstance(alphas, (tuple, list)): 
        alphas = torch.tensor(alphas, dtype=input.dtype, device=input.device)
    elif isinstance(alphas, torch.Tensor): 
        alphas = alphas.type_as(input).to(input.device)
    else:
        raise RuntimeError(f"Incorrect alphas type: {type(alphas)}. Alphas values {alphas}")
    
    expect_n_alphas = 2 if fl_type == 'binary' else input.shape[1]
    if alphas.shape[0] != expect_n_alphas:
        raise ValueError(f"Invalid alphas shape. we expect {expect_n_alphas} alpha values. Got: {alphas.shape[0]}.")
    
    # Normalize alphas to sum up 1
    alphas.div_(alphas.sum())
    
    # Non weighted version of Focal Loss computation:
    if fl_type == 'multiclass':
        target = target.long() # Targets needs to be long
        base_loss = F.cross_entropy(input, target, reduction='none')
    else: # Target can't be long
        base_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        
    target = target.type(torch.long)
    at = alphas.gather(0, target.data.view(-1))
    if fl_type == 'multilabel': # Reshape 
        at = at.view(-1, len(alphas))
        
    pt = torch.exp(-base_loss)
    focal_loss = at*(1-pt)** gamma * base_loss
        
    if reduction == 'none': return focal_loss
    elif reduction == 'mean': return focal_loss.mean()
    elif reduction == 'sum': return focal_loss.sum()
    else: raise NotImplementedError("Invalid reduction mode: {}".format(reduction))      
        
class FocalLoss(nn.Module):
    """    
    Focal loss that support binary, multiclass or multilabel classification. See [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).
    This implementation is a non weighted version of Focal Loss in contrast of some implementations. See
    this [kaggle post](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/162035#904086).

    According to the paper, the Focal Loss for binary case is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
        
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    
    The `input` is expected to contain raw, unnormalized scores for each class. `input` has to be a one hot encoded Tensor of size 
    either :math:`(minibatch, C)` for multilabel or multiclass classification or :math:`(minibatch, )` for binary classification. 
    
    The `target` is expected to contain raw, unnormalized scores for each class. `target` has to be a one hot encoded Tensor of size 
    either :math:`(minibatch, C)` for multilabel classification or :math:`(minibatch, )` for binary or multiclass classification. 
    
    Args:
        alphas (float, list, tuple, Tensor): the `alpha` value for each class. It weights the losses of each class. When `fl_type`
            is 'binary', it could be a float. In this case, it's transformed to :math:`alphas = (alphas, 1 - alphas)` where the
            first position is for the negative class and the second the positive. Note: alpha values are normalized to sum up 1.
        gamma (float): gamma exponent of the focal loss. Typically, between 0.25 and 4.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.
            
    
    :math:`(minibatch, C, d_1, d_2, ..., d_K)`
    
    Note: the implementation is based on [this post](https://amaarora.github.io/2020/06/29/FocalLoss.html).
    
    """
    def __init__(self, fl_type: str, alphas: Union[float, List[float]], gamma: float = 2.0, reduction: str = 'mean') -> None:
        super(FocalLoss, self).__init__()
        self.fl_type = fl_type
        self.alphas = alphas #torch.tensor(alphas)        
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, input, target):
        return focal_loss(input, target, fl_type=self.fl_type, alphas=self.alphas, gamma=self.gamma, reduction=self.reduction)        


def LossFunction(loss_function_name):
    if loss_function_name == 'L1Loss':
        return nn.L1Loss()
    elif loss_function_name == 'MSELoss':
        return nn.MSELoss()
    elif loss_function_name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif loss_function_name == 'CTCLoss':
        return nn.CTCLoss()
    elif loss_function_name == 'NLLLoss':
        return nn.NLLLoss()
    elif loss_function_name == 'PoissonNLLLoss':
        return nn.PoissonNLLLoss()
    elif loss_function_name == 'GaussianNLLLoss':
        return nn.GaussianNLLLoss()
    elif loss_function_name == 'KLDivLoss':
        return nn.KLDivLoss()
    elif loss_function_name == 'BCELoss':
        return nn.BCELoss(size_average=True)
    elif loss_function_name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss()
    elif loss_function_name == 'MarginRankingLoss':
        return nn.MarginRankingLoss()
    elif loss_function_name == 'HingeEmbeddingLoss':
        return nn.HingeEmbeddingLoss()
    elif loss_function_name == 'MultiLabelMarginLoss':
        return nn.MultiLabelMarginLoss()
    elif loss_function_name == 'HuberLoss':
        return nn.HuberLoss()
    elif loss_function_name == 'SmoothL1Loss':
        return nn.SmoothL1Loss()
    elif loss_function_name == 'SoftMarginLoss':
        return nn.SoftMarginLoss()
    elif loss_function_name == 'MultiLabelSoftMarginLoss':
        return nn.MultiLabelSoftMarginLoss()
    elif loss_function_name == 'CosineEmbeddingLoss':
        return nn.CosineEmbeddingLoss()
    elif loss_function_name == 'MultiMarginLoss':
        return nn.MultiMarginLoss()
    elif loss_function_name == 'TripletMarginLoss':
        return nn.TripletMarginLoss()
    elif loss_function_name == 'TripletMarginWithDistanceLoss':
        return nn.TripletMarginWithDistanceLoss()
    elif loss_function_name == 'FocalLoss':
        alpha_list = list(np.loadtxt("/data/thesis/dataset/alpha.csv", delimiter=','))
        return FocalLoss('multilabel', alphas=alpha_list, reduction='mean')
    elif loss_function_name == 'FocalLoss005':
        return Focalloss(alpha=0.05)
    elif loss_function_name == 'FocalLoss01':
        return Focalloss(alpha=0.1)
    elif loss_function_name == 'FocalLoss015':
        return Focalloss(alpha=0.15)
    else:
        return nn.CrossEntropyLoss()
# class LossFunction(nn.Module):
#     def __init__(self, loss_function_name):
#         super(LossFunction, self).__init__()
#         if loss_function_name == 'L1Loss':
#             self.loss_function = nn.L1Loss()
#         elif loss_function_name == 'MSELoss':
#             self.loss_function = nn.MSELoss()
#         elif loss_function_name == 'CrossEntropyLoss':
#             self.loss_function = nn.CrossEntropyLoss()
#         elif loss_function_name == 'CTCLoss':
#             self.loss_function = nn.CTCLoss()
#         elif loss_function_name == 'NLLLoss':
#             self.loss_function = nn.NLLLoss()
#         elif loss_function_name == 'PoissonNLLLoss':
#             self.loss_function = nn.PoissonNLLLoss()
#         elif loss_function_name == 'GaussianNLLLoss':
#             self.loss_function = nn.GaussianNLLLoss()
#         elif loss_function_name == 'KLDivLoss':
#             self.loss_function = nn.KLDivLoss()
#         elif loss_function_name == 'BCELoss':
#             self.loss_function = nn.BCELoss()
#         elif loss_function_name == 'BCEWithLogitsLoss':
#             self.loss_function = nn.BCEWithLogitsLoss()
#         elif loss_function_name == 'MarginRankingLoss':
#             self.loss_function = nn.MarginRankingLoss()
#         elif loss_function_name == 'HingeEmbeddingLoss':
#             self.loss_function = nn.HingeEmbeddingLoss()
#         elif loss_function_name == 'MultiLabelMarginLoss':
#             self.loss_function = nn.MultiLabelMarginLoss()
#         elif loss_function_name == 'HuberLoss':
#             self.loss_function = nn.HuberLoss()
#         elif loss_function_name == 'SmoothL1Loss':
#             self.loss_function = nn.SmoothL1Loss()
#         elif loss_function_name == 'SoftMarginLoss':
#             self.loss_function = nn.SoftMarginLoss()
#         elif loss_function_name == 'MultiLabelSoftMarginLoss':
#             self.loss_function = nn.MultiLabelSoftMarginLoss()
#         elif loss_function_name == 'CosineEmbeddingLoss':
#             self.loss_function = nn.CosineEmbeddingLoss()
#         elif loss_function_name == 'MultiMarginLoss':
#             self.loss_function = nn.MultiMarginLoss()
#         elif loss_function_name == 'TripletMarginLoss':
#             self.loss_function = nn.TripletMarginLoss()
#         elif loss_function_name == 'TripletMarginWithDistanceLoss':
#             self.loss_function = nn.TripletMarginWithDistanceLoss()
#         else:
#             self.loss_function = nn.CrossEntropyLoss()
#         print("Setup loss function done: "+loss_function_name+"!")
#     def forward(self):
#         return self.loss_function