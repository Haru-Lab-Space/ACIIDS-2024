# from icdmappings import Mapper
# import icd10
from tqdm import tqdm
import pandas as pd
from torch import nn
import numpy as np
from ast import literal_eval
import os
from utils.utils import write_json, read_json

class Tokenizer:
    def __init__(self):
        self.df_dict = None
        self.length = 0
    def build(self, path, tokenizer_path, category_map_path=None, category_type_map_path=None):
        if os.path.isfile(tokenizer_path):
            self.load(tokenizer_path)
            print("Loading tokenizer done!")
        else:
            self.load_df(path)
            self.save(tokenizer_path)
            print("Generating tokenizer done!")

        self.length = len(self.df_dict)

        if category_map_path != None:
            self.category_map = np.loadtxt(category_map_path, delimiter=',')
        if category_type_map_path != None:
            self.category_type_map = np.loadtxt(category_type_map_path, delimiter=',')
        
    def __len__(self):
        return self.length
    def keys(self):
        return self.df_dict.keys()
    def description(self, *args):
        if len(args) > 1:
            return self.df_dict[args].get('long_title')
        return self.df_dict[self.decode(args[0])].get('long_title')
    def add_encode(self, icd_code, icd_version, code):
        self.df_dict[icd_code, icd_version]['encode'] = code
    def __encode_step(self, icd_code, icd_version):
        return self.df_dict[icd_code, icd_version]['encode']
    def encode(self, icd_code, icd_version=None):
        if icd_version == None:
            if isinstance(icd_code, list):
                encode_list = []
                for code in icd_code:
                    encode_list.append(self.__encode_step(code['icd_code'], code['icd_version']))
                return encode_list
        else:
            if isinstance(icd_code, list):
                encode_list = []
                for i in range(len(icd_code)):
                    encode_list.append(self.__encode_step(icd_code[i], icd_version[i]))
                return encode_list
            else:
                return self.__encode_step(icd_code, icd_version)
    def __binary_encode_step(self, icd_code, icd_version):
        return self.df_dict[icd_code, icd_version]['binary_encode']
    # def binary_encode(self, icd_code, icd_version=None, top_k_dataset=100000):
    def binary_encode(self, icd_code, icd_version=None, top_k_dataset=None, category_map_state=False, category_type_map_state=False):
        if icd_version == None:
            # Mã gửi vào là danh sách encode và ta muốn tạo binary của encode đó
            if isinstance(icd_code, list):
                bi_encode = [0.0] * self.length
                if top_k_dataset == None:
                    for i in range(len(icd_code)):
                    # for i in range(min(top_k_dataset, len(icd_code))):
                        bi_encode[icd_code[i]] = 1.0
                else:
                    for i in range(min(top_k_dataset, len(icd_code))):
                        bi_encode[icd_code[i]] = 1.0

                if category_map_state:
                    bi_encode = np.array(bi_encode).reshape(1, -1)
                    bi_encode = np.matmul(bi_encode, self.category_map).reshape(-1)
                    return bi_encode
                elif category_type_map_state:
                    bi_encode = np.array(bi_encode).reshape(1, -1)
                    bi_encode = np.matmul(bi_encode, self.category_type_map).reshape(-1)
                    return bi_encode
                else:
                    return bi_encode

            # Mã gửi vào là một encode
            else:
                temp = self.decode(icd_code)
                return self.__binary_encode_step(temp['icd_code'], temp['icd_version'])
        else:
            # Mã gửi vào là danh sách icd_code, icd_version
            if isinstance(icd_code, list):
                encode_list = [0.0] * self.length
                for i in range(len(icd_code)):
                # for i in range(min(top_k_dataset, len(icd_code))):
                    encode_list = np.logical_or(encode_list, self.__binary_encode_step(icd_code[i], icd_version[i]))
                return encode_list
            # Mã gửi vào là một icd_code - icd_version
            else:
                return self.__binary_encode_step(icd_code, icd_version)

    def decode(self, encode):
        for key, value in self.df_dict.items():
            if encode == value['encode']:
                return {'icd_code': key[0],
                        'icd_version': key[1]}
    def binary_decode(self, binary_encode, top_k):
        decode_list = []
        for i in range(len(binary_encode)):
            if len(decode_list) >= top_k:
                break
            if binary_encode[i] == 1.0:
                decode_list.append(self.decode(i))
        return decode_list


    def load_df(self, df_path):
        df = pd.read_csv(df_path)
        self.df_dict = df.set_index(['icd_code', 'icd_version']).to_dict(orient='index')
        self.auto_encode()
        # self.auto_binary_encode()
    def load(self, file_path):
        # Load the dictionary from the JSON file
        obj = read_json(file_path)
        self.df_dict = {literal_eval(k): v for k, v in obj.items()}
    def save(self, file_path):
        # Save the dictionary to a JSON file
        dict = {str(k): v for k, v in self.df_dict.items()}
        print("Save tokenizer vocab")
        write_json(dict, file_path)
    
    def auto_encode(self):
        i = 0 
        logger_message = f'Auto encode '
        progress_bar = tqdm(self.df_dict,
                            desc=logger_message, initial=0, dynamic_ncols=True)
        for key in self.df_dict:
            self.df_dict[key]['encode'] = i
            i+=1
            progress_bar.update()
        progress_bar.close()
    def auto_binary_encode(self):
        logger_message = f'Auto binary encode '
        progress_bar = tqdm(self.df_dict,
                                desc=logger_message, initial=0, dynamic_ncols=True)
        for key in self.df_dict:
            sample = [0.0] * self.length
            encode = self.df_dict[key]['encode']
            sample[encode] = 1.0
            self.df_dict[key]['binary_encode'] = sample
            progress_bar.update()
        progress_bar.close()