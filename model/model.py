import torch
from torch import nn, Tensor
import numpy as np
import time
import json
def write_json(dict, parient_dir, name=None):
    # Serializing json
    json_object = json.dumps(dict, indent=4)
    if name != None:
        path = os.path.join(parient_dir, str(name)+".json")
    else:
        path = parient_dir
    # Writing to sample.json
    with open(path, "w") as outfile:
        outfile.write(json_object)
class TTT_Time_Ensemble_Multiscale_Decoder_Only_1(nn.Module):
    def __init__(self, input_size, category_size, category_type_size, hidden_size, time_hidden_size, key_query_hidden_size, key_time_hidden_size, dropout):
        """The __init__ method that initiates an RNN instance.

        Args:
            input_dim (int): The number of nodes in the input layer
            hidden_dim (int): The number of nodes in each layer
            layer_dim (int): The number of layers in the network
            output_dim (int): The number of nodes in the output layer
            dropout_prob (float): The probability of nodes being dropped out

        """
        super(TTT_Time_Ensemble_Multiscale_Decoder_Only_1, self).__init__()
        self.key_query_hidden_size = key_query_hidden_size

        self.visit_feedforward = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.category_feedforward = nn.Linear(in_features=category_size, out_features=hidden_size)
        self.category_type_feedforward = nn.Linear(in_features=category_type_size, out_features=hidden_size)
        # self.visit_projection = nn.Linear(in_features=input_size, out_features=1)
        # self.category_projection = nn.Linear(in_features=category_size, out_features=1)

        self.time_stage_1_embedding = nn.Linear(1, time_hidden_size)
        self.time_stage_2_embedding = nn.Linear(time_hidden_size, hidden_size)

        # self.time_project_1 = nn.Linear(1, category_size)
        # self.time_project_2 = nn.Linear(1, input_size)


        # visit_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, batch_first=True, dropout=dropout)
        # self.visit_transformer_encoder = nn.TransformerEncoder(visit_encoder_layer, num_layers=1)

        
        category_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, batch_first=True, dropout=dropout)
        self.category_transformer_encoder = nn.TransformerEncoder(category_encoder_layer, num_layers=1)


        # category_type_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, batch_first=True, dropout=dropout)
        # self.category_type_transformer_encoder = nn.TransformerEncoder(category_type_encoder_layer, num_layers=1)

        self.category_intergrate = nn.Linear(in_features=2, out_features=1)
        self.category_type_intergrate = nn.Linear(in_features=2, out_features=1)
        self.visit_intergrate = nn.Linear(in_features=3, out_features=1)
        self.feature_feedforward = nn.Linear(in_features=3*hidden_size, out_features=1)

        self.ReLU = nn.ReLU()
        self.key_query_layer = nn.Linear(3*hidden_size, key_query_hidden_size)
        self.key_time_stage_1_embedding = nn.Linear(1, key_time_hidden_size)
        self.key_time_stage_2_embedding = nn.Linear(key_time_hidden_size, key_query_hidden_size)

        self.fusion_layer = nn.Linear(3*hidden_size, 2)
        self.fc = nn.Linear(3*hidden_size, input_size)
    def __forward_step(self, input_visits, overall_visits, input_visit_categories, input_overall_visit_categories, input_visit_category_type, input_overall_visit_category_type, reverse_time):
        """The forward method takes input tensor x and does forward propagation

        Args:
            x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

        Returns:
            torch.Tensor: The output tensor of the shape (batch size, output_dim)

        """
        # input_visits : torch.Size([2, 1, 512])
        # overall_visits : torch.Size([2, 512])
        # input_visit_categories : torch.Size([2, 1, 543])
        # input_overall_visit_categories : torch.Size([2, 543])
        # reverse_time : torch.Size([2, 1])
        overall_reverse_time = reverse_time.min(dim=1).values.unsqueeze(1)
        overall_reverse_time = torch.cat((reverse_time, overall_reverse_time), dim = 1).unsqueeze(-1)

        input_visits = torch.cat((input_visits, overall_visits.unsqueeze(1)), dim=1)
        input_visit_categories = torch.cat((input_visit_categories, input_overall_visit_categories.unsqueeze(1)), dim=1)
        input_visit_category_type = torch.cat((input_visit_category_type, input_overall_visit_category_type.unsqueeze(1)), dim=1)


        input_visits = self.visit_feedforward(input_visits) # torch.Size([2, 63, 64])
        input_visit_categories = self.category_feedforward(input_visit_categories) # torch.Size([2, 63, 64])
        input_visit_category_type = self.category_type_feedforward(input_visit_category_type) # torch.Size([2, 63, 64])

        time_vector = self.time_stage_2_embedding(1-torch.tanh(torch.pow(self.time_stage_1_embedding(overall_reverse_time/180), 2)))
        input_visit_category_type = self.category_type_intergrate(torch.cat([input_visit_category_type.unsqueeze(-1), time_vector.unsqueeze(-1)], dim = -1)).squeeze(-1)
        input_visit_category_type = self.category_transformer_encoder(input_visit_category_type)
        input_visit_categories = self.category_intergrate(torch.cat([input_visit_categories.unsqueeze(-1), input_visit_category_type.unsqueeze(-1)], dim = -1)).squeeze(-1)
        input_visit_categories = self.category_transformer_encoder(input_visit_categories)
        info = self.visit_intergrate(torch.cat([input_visits.unsqueeze(-1), input_visit_categories.unsqueeze(-1), input_visit_category_type.unsqueeze(-1)], dim=-1)).squeeze(-1)
        info = self.category_transformer_encoder(info)

        final_feature = torch.cat([input_visit_categories, input_visit_category_type, info], dim=-1)
        hidden_feature, last_hidden_feature = final_feature[:, :-1], final_feature[:, -1]

        # feature_attn = self.feature_feedforward(hidden_feature)
        # final_feature = torch.matmul(feature_attn.permute(0,2,1), hidden_feature).squeeze(1)

        alpha = torch.softmax(self.feature_feedforward(hidden_feature), axis=1).transpose(-2, -1)

        # Global Level: Comprehensive Analysis
        q = self.ReLU(self.key_query_layer(last_hidden_feature)).unsqueeze(1)
        k = torch.tanh(self.key_time_stage_2_embedding(1 - torch.tanh(torch.pow(self.key_time_stage_1_embedding(reverse_time.unsqueeze(-1)/180), 2))))
        
        theta = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.key_query_hidden_size)
        beta = torch.softmax(theta, dim = -1)

        # Time-aware Dynamic Attention Fusion
        fusion_attention = torch.softmax(self.fusion_layer(last_hidden_feature), dim=-1).unsqueeze(1)
        overall_attention_weight = torch.matmul(fusion_attention, torch.cat([alpha, beta], dim = 1))
        h_hat = torch.matmul(overall_attention_weight / torch.sum(overall_attention_weight, dim =-1).unsqueeze(1), hidden_feature).squeeze(1)

        score = overall_attention_weight / torch.sum(overall_attention_weight, dim =-1).unsqueeze(1)
        # write_json({"overall": score.detach().cpu().tolist(),
        #             "local attention": alpha.detach().cpu().tolist(),
        #             "global attention": beta.detach().cpu().tolist()}, "/data/thesis/save_directory/interpreting_directory/only_1_score.json")
        return self.fc(h_hat)
    
    def forward(self, x):
        visits = x['visits']
        visit_categories = x['visit_categories']
        visit_category_type = x['visit_category_type']
        label = x['labels']
        reverse_time = x['reverse_time']
        batch_size = visits.shape[0]
        max_len = visits.shape[1]
        num_label = visits.shape[2]
        num_categories = visit_categories.shape[2]
        num_category_type = visit_category_type.shape[2]
        input_overall_visits = torch.zeros((batch_size, num_label)).to(visits.get_device())
        input_overall_visit_categories = torch.zeros((batch_size, num_categories)).to(visits.get_device())
        input_overall_input_visit_category_type = torch.zeros((batch_size, num_category_type)).to(visits.get_device())
        preds_list = None
        target_list = None
        for i in range(max_len - 1, 0, -1):
            input_visits, _ = visits[:, :-i], visits[:, -i]
            input_visit_categories, _ = visit_categories[:, :-i], visit_categories[:, -i]
            input_visit_category_type, _ = visit_category_type[:, :-i], visit_categories[:, -i]
            targets = label[:, -i-1]
            input_reverse_time = reverse_time[:, :-i] - reverse_time[:, -i].unsqueeze(1)
            input_overall_visits = torch.logical_or(input_overall_visits, visits[:, -i-1])
            input_overall_visit_categories = torch.logical_or(input_overall_visit_categories, visit_categories[:, -i-1])
            input_overall_input_visit_category_type = torch.logical_or(input_overall_input_visit_category_type, visit_category_type[:, -i-1])
            
            preds = self.__forward_step(input_visits, input_overall_visits, input_visit_categories, input_overall_visit_categories, input_visit_category_type, input_overall_input_visit_category_type, input_reverse_time)
            if preds_list == None:
                preds_list = preds
                target_list = targets
            else:
                preds_list = torch.cat((preds_list, preds), 0)
                target_list = torch.cat((target_list, targets), 0)
        
        return preds_list, target_list