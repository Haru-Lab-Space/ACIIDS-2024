import pandas as pd
from datasets import Dataset, DatasetDict
from torch import nn
import datetime
import torch
import numpy as np

class DataGenerator(nn.Module):
    def __init__(self, file_path, tokenizer, columns=None, min_appearances=3, max_appearances=64, validation_size=0.2):
        super(DataGenerator, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_appearances
        self.columns = columns
        self.unknown_sample = [False] * len(self.tokenizer.df_dict)
        self.count = 0


        df = pd.read_csv(file_path).sort_values(['subject_id','admittime','seq_num'])
        print("Loading data done!")

        df = df[["subject_id", 'gender', 'anchor_age', 'anchor_year', 'hadm_id', 'icd_code',
            'icd_version', 'admittime',
            'admission_type', 'admission_location',
            'discharge_location', 'insurance', 'language', 'marital_status', 'race']]
        df['encode'] = df.apply(lambda x: tokenizer.encode(x['icd_code'], x['icd_version']), axis=1)
        df['binary_encode'] = df.apply(lambda x: tokenizer.binary_encode(x['icd_code'], x['icd_version']), axis=1)
        print("Encoding done!")

        grouped = df.groupby(["subject_id", 'gender', 'anchor_age', 'anchor_year', 'hadm_id', 'admittime',
            'admission_type', 'admission_location',
            'discharge_location', 'insurance', 'language', 'marital_status', 'race'], dropna=False)
        merged_data = grouped.agg({
            "icd_code": lambda x: list(x),
            "icd_version": lambda x: list(x),
            "encode": lambda x: list(x),
            "binary_encode": lambda x: list(x),
        }).reset_index().sort_values(['subject_id','admittime'])
        value_counts = merged_data['subject_id'].value_counts()

        # Filter the DataFrame to keep only values that appear 3 or more times
        filtered_df = merged_data[merged_data['subject_id'].isin(value_counts[value_counts >= min_appearances].index)].sort_values(['subject_id','admittime'])
        filtered_df = filtered_df[filtered_df['subject_id'].isin(value_counts[value_counts <= max_appearances].index)].sort_values(['subject_id','admittime'])
        
        merged_data = filtered_df
        print("Merging icd-codes done!")
        merged_data['binary_visits'] = merged_data.apply(lambda x: self.binary_encode(x), axis=1)
        print("Binary visit done!")

        merged_data['date'] = pd.to_datetime(merged_data['admittime']).dt.date
        merged_data['month'] = pd.to_datetime(merged_data['admittime']).dt.month
        merged_data['year'] = pd.to_datetime(merged_data['admittime']).dt.year
        merged_data["lag_year"] = merged_data["year"] - merged_data['anchor_year']
        
        merged_data['datetime_month_year'] = pd.to_datetime(merged_data[['year', 'month']].assign(day=1))
        merged_data['month_year'] = merged_data["year"] * 12 + merged_data['month']
        merged_data['lag_month_year'] = merged_data['month_year'] - merged_data["anchor_year"] * 12

        print("Setting time done!")
        merged_data=merged_data.sort_values(['subject_id','admittime'])
        grouped = merged_data.groupby(["subject_id", 'gender', 'anchor_age', 'anchor_year'], dropna=False)
        self.merged_data = grouped.agg({
            "hadm_id": lambda x: list(x),
            "icd_code": lambda x: list(x),
            "icd_version": lambda x: list(x),
            "encode": lambda x: list(x),
            "binary_encode": lambda x: list(x),
            "binary_visits": lambda x: list(x),
            "admittime": lambda x: list(x),
            "year": lambda x: list(x),
            "month": lambda x: list(x),
            "date": lambda x: list(x),
            "lag_year": lambda x: list(x),
            "month_year": lambda x: list(x),
            "datetime_month_year": lambda x: list(x),
            "lag_month_year": lambda x: list(x),
            "admission_type": lambda x: list(x),
            "admission_location": lambda x: list(x),
            "discharge_location": lambda x: list(x),
            "marital_status": lambda x: list(x),
            "insurance": lambda x: list(x),
            "race": 'last',
            "language": 'last',
        }).reset_index().sort_values(['subject_id'])

        self.merged_data = self.merged_data.apply(lambda x: self.padding_visits(x))
        # self.merged_data['binary_visits'] = torch.tensor(self.merged_data['binary_visits'], dtype= torch.float)
        print("Merging visits done!")
        if columns != None:
            self.merged_data = self.merged_data[columns]
            print("Loading optional columns: "+str(columns)+"!")

        n = len(self.merged_data)
        n_train = int(n*(1-validation_size))
        self.dataset = DatasetDict({'train': Dataset.from_pandas(self.merged_data[:n_train], preserve_index=False),
                                    'validation': Dataset.from_pandas(self.merged_data[n_train:], preserve_index=False),
                                    })
    def forward(self):
        return self.dataset
    def train_test_split(self, test_size):
        return self.dataset.train_test_split(test_size=test_size).values()
    def binary_encode(self, sample):
        # print(self.count)
        # self.count+=1
        return np.logical_or.reduce(sample['binary_encode'])
    def padding_visits(self, sample):
        while(len(sample['binary_visits']) < self.max_len):
            sample['binary_visits'].insert(0, self.unknown_sample)
            sample['lag_month_year'].insert(0, 0)
            sample['marital_status'].insert(0, 0)
            sample['insurance'].insert(0, 0)
            sample['race'].insert(0, 0)
            sample['language'].insert(0, 0)
        return sample
    def dataframe(self):
        return self.merged_data
    
    # def __len__(self):
    #     return len(self.landmarks_frame)

    # def __getitem__(self, idx):
    #     if torch.is_tensor(idx):
    #         idx = idx.tolist()

    #     sample = {}
    #     sample = {'image': image, 'landmarks': landmarks}

    #     return sample