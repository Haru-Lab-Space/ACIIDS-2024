import pandas as pd
from datasets import Dataset, DatasetDict
from torch import nn
import datetime
import torch
import numpy as np
from utils.utils import read_json, write_json

class CustomDataset(Dataset):
    def __init__(self, parient_dir, keys, max_len,split=None, tokenizer=None, top_k_evaluate=None, opt=None):
        self.objects = read_json(parient_dir, split)
        self.keys = keys
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.top_k_evaluate = top_k_evaluate
        self.category_map_state = opt.category_map_state
        self.category_size = opt.category_size
        self.category_type_size = opt.category_type_size
        self.category_type_map_state = opt.category_type_map_state
        print("category_map_state: "+str(self.category_map_state))
        print("category_type_map_state: "+str(self.category_type_map_state))
        self.opt = opt
    def __getitem__(self, index):
        dict = {}
        for key in self.keys:
            dict[key] = self.padding(key, self.objects[key][str(index)])
        
        dict['visits'] = self.initialize_visit(self.objects['encode'][str(index)])
        dict['labels'] = self.initialize_label(self.objects['encode'][str(index)])
        if self.category_map_state:
            dict['visit_categories'] = self.initialize_visit_categories(self.objects['encode'][str(index)])
        if self.category_type_map_state:
            dict['visit_category_type'] = self.initialize_visit_category_type(self.objects['encode'][str(index)])
        # if 'reverse_time' in self.keys:
        #     dict['overall_visits'] = np.logical_or.reduce(dict['visits'], axis=1).unsqueeze(1)
        return dict

    def __len__(self):
        return len(self.objects['subject_id'])
    
    def initialize_visit(self, encode_sample):
        array = np.zeros(shape=(self.max_len, len(self.tokenizer)), dtype = 'f')
        internal = self.max_len - len(encode_sample)
        for index_visit in range(len(encode_sample) - 1):
            array[internal + index_visit] = self.tokenizer.binary_encode(encode_sample[index_visit])
        return array

    def initialize_visit_categories(self, encode_sample):
        array = np.zeros(shape=(self.max_len, self.category_size), dtype = 'f')
        internal = self.max_len - len(encode_sample)
        for index_visit in range(len(encode_sample) - 1):
            array[internal + index_visit] = self.tokenizer.binary_encode(encode_sample[index_visit], category_map_state = self.category_map_state)

        return array

    
    def initialize_visit_category_type(self, encode_sample):
        array = np.zeros(shape=(self.max_len, self.category_type_size), dtype = 'f')
        internal = self.max_len - len(encode_sample)
        for index_visit in range(len(encode_sample) - 1):
            array[internal + index_visit] = self.tokenizer.binary_encode(encode_sample[index_visit], category_type_map_state = self.category_type_map_state)

        return array
    
    def initialize_label(self, encode_sample):
        array = np.zeros(shape=(self.max_len, len(self.tokenizer)), dtype = 'f')
        internal = self.max_len - len(encode_sample)
        for index_visit in range(len(encode_sample)-1):
            if self.top_k_evaluate != None:
                array[internal + index_visit] = self.tokenizer.binary_encode(icd_code=encode_sample[index_visit+1], top_k_dataset=self.top_k_evaluate)
            else:
                array[internal + index_visit] = self.tokenizer.binary_encode(encode_sample[index_visit+1])
        # if self.top_k_evaluate != None:
        #     array[- 1] = self.tokenizer.binary_encode(icd_code=encode_sample[index_visit], top_k_dataset=self.top_k_evaluate)
        return array
    
    def padding(self, name_key, obj):
        if name_key in ['gender', 'insurance', 'language', 'marital_status', 'race']:
            key_dim = len(obj[0])
            array = np.zeros(shape=(self.max_len, key_dim), dtype = 'f')
            internal = self.max_len - len(obj)
            for index_visit in range(len(obj)):
                array[internal + index_visit] = obj[index_visit]
        elif name_key in ['age', 'time', 'month', 'reverse_time', 'mask_attention']:
            array = np.zeros(shape=(self.max_len), dtype = 'f')
            internal = self.max_len - len(obj)
            for index_visit in range(len(obj)):
                array[internal + index_visit] = obj[index_visit]
        else:
            return obj
        return array
    def set_predicted_info(self, predicted_labels, true_labels):
        self.objects['predicted_labels'] = self.list_to_dict_with_index(predicted_labels)
        self.objects['true_labels'] = self.list_to_dict_with_index(true_labels)
    
    def save_predicted_info(self, predicted_dataset_directory, model_name):
        old_keys, new_keys = list(self.objects.keys())[:-2], list(self.objects.keys())[-2:]
        new_len = len(self.objects['predicted_labels'])
        old_len = len(self.objects['subject_id'])
        def remove_last_index(my_dict, l):
            for _ in range(l):
                my_dict.popitem()
            return my_dict
        for key in old_keys:
            l = (old_len-new_len)
            self.objects[key] = remove_last_index(self.objects[key], l)
        write_json(self.objects, predicted_dataset_directory, model_name)
        
    def list_to_dict_with_index(self, lst):
        dict_with_index = {index: value for index, value in enumerate(lst)}
        return dict_with_index