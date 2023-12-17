from torch import nn
import torch

def Optimizer(model, optimizer_name, learning_rate, **kwargs):
    if optimizer_name == 'Adadelta':
        return torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'Adagrad':
        return torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SparseAdam':
        return torch.optim.SparseAdam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'Adamax':
        return torch.optim.Adamax(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'ASGD':
        return torch.optim.ASGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'LBFGS':
        return torch.optim.LBFGS(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'NAdam':
        return torch.optim.NAdam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'RAdam':
        return torch.optim.RAdam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'RMSprop':
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'Rprop':
        return torch.optim.Rprop(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=kwargs.get('momentum', 0))
    else:
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
# class Optimizer:
#     def __init__(self, model, optimizer_name, learning_rate, **kwargs):
#         if optimizer_name == 'Adadelta':
#             self.optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
#         elif optimizer_name == 'Adagrad':
#             self.optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
#         elif optimizer_name == 'Adam':
#             self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#         elif optimizer_name == 'AdamW':
#             self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#         elif optimizer_name == 'SparseAdam':
#             self.optimizer = torch.optim.SparseAdam(model.parameters(), lr=learning_rate)
#         elif optimizer_name == 'Adamax':
#             self.optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
#         elif optimizer_name == 'ASGD':
#             self.optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate)
#         elif optimizer_name == 'LBFGS':
#             self.optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate)
#         elif optimizer_name == 'NAdam':
#             self.optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)
#         elif optimizer_name == 'RAdam':
#             self.optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)
#         elif optimizer_name == 'RMSprop':
#             self.optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
#         elif optimizer_name == 'Rprop':
#             self.optimizer = torch.optim.Rprop(model.parameters(), lr=learning_rate)
#         elif optimizer_name == 'SGD':
#             self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=kwargs.get('momentum', 0))
#         print("Setup optimizer done: "+optimizer_name+"!")
#     def forward(self):
#         return self.optimizer