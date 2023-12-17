import torch
from torch import nn

class BackBone(nn.Module):
    def __init__(self, base_model, head=None, **kwargs):
        super(BackBone, self).__init__()
        self.model = nn.Sequential()
        self.model.base = base_model
        if head != None:
            self.model.head = head
    def forward(self, x, **kwargs):
        _x = self.model(x)
        return _x