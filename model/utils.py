import torch
from torch import nn

"""
import models
"""
from model import TTT_Time_Ensemble_Multiscale_Decoder_Only_1

def Model(model_name, opt):
    if model_name == 'TTT_Time_Ensemble_Multiscale_Decoder_Only_1':
        return TTT_Time_Ensemble_Multiscale_Decoder_Only_1(input_size=opt.input_size, 
                       category_size=opt.category_size,
                       category_type_size=opt.category_type_size,
                       hidden_size=opt.HiTANet_hidden_size, 
                       time_hidden_size=opt.HiTANet_time_hidden_size, 
                       key_query_hidden_size=opt.HiTANet_key_query_hidden_size, 
                       key_time_hidden_size=opt.HiTANet_key_time_hidden_size,
                       dropout=opt.HiTANet_dropout)