import os
import numpy as np
import pandas as pd
import json
from json import loads, dumps
import torch
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from ast import literal_eval
import time
import swifter
import shutil

CHECKPOINT_EXTENSION = '.pt'
MODEL_EXTENSION = '.pt'

    
def print_config(opt, model, loss_fn=None, optimizer=None):
    if loss_fn != None:
        print("Optimizer: " + str(optimizer))
    if optimizer != None:
        print("Loss function:" + str(loss_fn))
    print("____________________________")
    print("Model: " + str(model.__class__.__name__))
    print("Architure:")
    print(model)
    print("____________________________")
    time.sleep(10)
    
    print("Batch size: " +str(opt.batch_size))
    print("Top k medical codes: "+str(opt.top_k))
    print("Save folder: "+str(opt.save_directory))
    print("Dataset folder: "+str(opt.dataset_directory))
    print("device: "+str(opt.device))
    print("____________________________")
    print("Num labels: "+str(opt.input_size))
    print("Num ancestors: "+str(opt.num_ancestor))
    print("Learning rate: "+str(opt.learning_rate))
    print("____________________________")
    print("Have ancestors: "+str(opt.have_ancestor))
    print("Have times: "+str(opt.have_time))
    print("____________________________")
    print("num_workers: "+str(opt.num_workers))
    print("pin_memory: "+str(opt.pin_memory))
    print("shuffle: "+str(opt.shuffle))
    print("With clip: " +str(opt.with_clip))
    print("patience: " +str(opt.patience))
    print("min_delta: " +str(opt.min_delta))
    print("co_map: " +str(opt.co_map))
    print("____________________________")
    print("____________________________")

def makedir(parent_dir, directory):
    # Create the directory
    path = os.path.join(parent_dir, directory)
    
    try:
        os.makedirs(path, exist_ok = False)
        print("Directory '%s' created successfully" % directory)
    except OSError as error:
        print("Directory '%s' can not be created" % directory)

def ccsr_process(opt):
    def auto_encode(df, df_dict):
        i = 0 
        logger_message = f'Auto encode '
        progress_bar = tqdm(df_dict,
                            desc=logger_message, initial=0, dynamic_ncols=True)
        for key in df_dict:
            df_dict[key] = i
            df.replace(key, i, inplace=True)
            i+=1
            progress_bar.update()
        progress_bar.close()
        return df, df_dict
    
    def create_matrix(df, df_dict, attribute):
        icd_code = len(df)
        categories = len(df_dict)
        print("categories: "+str(categories))
        print("icd_code: "+str(icd_code))
        matrix = np.zeros((icd_code, categories), dtype=bool)
        for i in range(len(df)):
            list = df.iloc[i][attribute]
            for category in list:
                matrix[i][category] = True
        return matrix
            
        
    d_code_dataset = pd.read_csv(opt.d_code_dataset_path)
    DXCCSR_v2023 = pd.read_csv(opt.DXCCSR_v2023_path)
    DXCCSR_v2023 = DXCCSR_v2023.rename(columns={"\'ICD-10-CM CODE\'": 'ICD-10-CM CODE',
                                                "\'ICD-10-CM CODE DESCRIPTION\'": 'ICD-10-CM CODE DESCRIPTION',
                                                "\'Default CCSR CATEGORY IP\'": 'Default CCSR CATEGORY IP',
                                                "\'Default CCSR CATEGORY DESCRIPTION IP\'": 'Default CCSR CATEGORY DESCRIPTION IP',
                                                "\'Default CCSR CATEGORY OP\'": 'Default CCSR CATEGORY OP',
                                                "\'Default CCSR CATEGORY DESCRIPTION OP\'": 'Default CCSR CATEGORY DESCRIPTION OP',
                                                "\'CCSR CATEGORY 1\'": 'CCSR CATEGORY 1',
                                                "\'CCSR CATEGORY 1 DESCRIPTION\'": 'CCSR CATEGORY 1 DESCRIPTION',
                                                "\'CCSR CATEGORY 2\'": 'CCSR CATEGORY 2',
                                                "\'CCSR CATEGORY 2 DESCRIPTION\'": 'CCSR CATEGORY 2 DESCRIPTION',
                                                "\'CCSR CATEGORY 3\'": 'CCSR CATEGORY 3',
                                                "\'CCSR CATEGORY 3 DESCRIPTION\'": 'CCSR CATEGORY 3 DESCRIPTION',
                                                "\'CCSR CATEGORY 4\'": 'CCSR CATEGORY 4',
                                                "\'CCSR CATEGORY 4 DESCRIPTION\'": 'CCSR CATEGORY 4 DESCRIPTION',
                                                "\'CCSR CATEGORY 5\'": 'CCSR CATEGORY 5',
                                                "\'CCSR CATEGORY 5 DESCRIPTION\'": 'CCSR CATEGORY 5 DESCRIPTION',
                                                "\'CCSR CATEGORY 6\'": 'CCSR CATEGORY 6',
                                                "\'CCSR CATEGORY 6 DESCRIPTION\'": 'CCSR CATEGORY 6 DESCRIPTION',})
    DXCCSR_v2023 = DXCCSR_v2023.applymap(lambda x: x.lstrip("\'") if isinstance(x, str) else x)
    DXCCSR_v2023 = DXCCSR_v2023.applymap(lambda x: x.rstrip("\'") if isinstance(x, str) else x)
    
    df = pd.merge(d_code_dataset, DXCCSR_v2023, right_on='ICD-10-CM CODE', left_on='icd_code', how='inner').sort_values(['CCSR CATEGORY 1', 
                                                                                                                            'CCSR CATEGORY 2',
                                                                                                                            'CCSR CATEGORY 3',
                                                                                                                            'CCSR CATEGORY 4',
                                                                                                                            'CCSR CATEGORY 5',
                                                                                                                            'CCSR CATEGORY 6'])
    df["CATEGORY 1"] = df["CCSR CATEGORY 1"].apply(lambda x: x[:3])
    df["CATEGORY 2"] = df["CCSR CATEGORY 2"].apply(lambda x: x[:3])
    df["CATEGORY 3"] = df["CCSR CATEGORY 3"].apply(lambda x: x[:3])
    df["CATEGORY 4"] = df["CCSR CATEGORY 4"].apply(lambda x: x[:3])
    df["CATEGORY 5"] = df["CCSR CATEGORY 5"].apply(lambda x: x[:3])
    df["CATEGORY 6"] = df["CCSR CATEGORY 6"].apply(lambda x: x[:3])
    a1 = list(df['CCSR CATEGORY 1'].drop_duplicates())
    a2 = list(df['CCSR CATEGORY 2'].drop_duplicates())
    a3 = list(df['CCSR CATEGORY 3'].drop_duplicates())
    a4 = list(df['CCSR CATEGORY 4'].drop_duplicates())
    a5 = list(df['CCSR CATEGORY 5'].drop_duplicates())
    a6 = list(df['CCSR CATEGORY 6'].drop_duplicates())
    c1 = list(df['CATEGORY 1'].drop_duplicates())
    c2 = list(df['CATEGORY 2'].drop_duplicates())
    c3 = list(df['CATEGORY 3'].drop_duplicates())
    c4 = list(df['CATEGORY 4'].drop_duplicates())
    c5 = list(df['CATEGORY 5'].drop_duplicates())
    c6 = list(df['CATEGORY 6'].drop_duplicates())
    keys_list = a1 + a2 + a3 + a4 + a5 + a6
    type_keys_list = c1 + c2 + c3 + c4 + c5 + c6
    keys_list = sorted(keys_list)
    type_keys_list = sorted(type_keys_list)
    sam = keys_list[0]
    keys_list = keys_list[5:]
    type_keys_list = type_keys_list[5:]
    my_dict = dict.fromkeys(keys_list)
    df, my_dict = auto_encode(df, my_dict)
    my_type_dict = dict.fromkeys(type_keys_list)
    df, my_type_dict = auto_encode(df, my_type_dict)

    df['CCSR CATEGORY'] = df.apply(lambda row: [row['CCSR CATEGORY 1'], row['CCSR CATEGORY 2'], row['CCSR CATEGORY 3'], row['CCSR CATEGORY 4'], row['CCSR CATEGORY 5'], row['CCSR CATEGORY 6']], axis=1)
    df['CCSR CATEGORY'] = df['CCSR CATEGORY'].apply(lambda x: [y for y in x if y != sam])
    df['CCSR CATEGORY TYPE'] = df.apply(lambda row: [row['CATEGORY 1'], row['CATEGORY 2'], row['CATEGORY 3'], row['CATEGORY 4'], row['CATEGORY 5'], row['CATEGORY 6']], axis=1)
    df['CCSR CATEGORY TYPE'] = df['CCSR CATEGORY TYPE'].apply(lambda x: [y for y in x if y != sam])
    df = df[['icd_code',
            'icd_version',
            'long_title',
            'CCSR CATEGORY',
            'CCSR CATEGORY TYPE']]
    df.to_csv(opt.ccsr_category_path, index=False)
    matrix = create_matrix(df, my_dict, attribute='CCSR CATEGORY')
    type_matrix = create_matrix(df, my_type_dict, attribute='CCSR CATEGORY TYPE')
    write_json(my_dict, opt.dataset_directory, "category_map")
    write_json(my_type_dict, opt.dataset_directory, "category_type_map")
    np.savetxt(opt.category_map_path, matrix, delimiter=',')
    print("category_map_path: "+str(opt.category_map_path))
    print("category_type_map_path: "+str(opt.category_type_map_path))
    np.savetxt(opt.category_type_map_path, type_matrix, delimiter=',')
    return len(my_dict), len(my_type_dict)

def all_processing(tokenizer, opt):
    diagnoses_icd = pd.read_csv(opt.diagnoses_icd)
    d_icd_diagnoses = pd.read_csv(opt.d_icd_diagnoses)
    patients = pd.read_csv(opt.patients)
    admissions = pd.read_csv(opt.admissions)
    DXCCSR_v2023 = pd.read_csv(opt.DXCCSR_v2023_path)
    diagnoses_icd = diagnoses_icd[diagnoses_icd['icd_version']==10]
    df = pd.merge(diagnoses_icd, patients, how='inner')
    df = pd.merge(df, admissions, how='inner')
    print("Original dataset: " + str(len(df)))
    
    temp_df = df.groupby(["subject_id", "hadm_id"]).first().reset_index()
    value_counts = temp_df['subject_id'].value_counts()
    # Filter the DataFrame to keep only visit values that appear not less than 3 and not larger than 64
    temp_df = temp_df[temp_df['subject_id'].isin(value_counts[value_counts >= opt.min_appearances].index)].sort_values(['subject_id','admittime'])
    temp_df = temp_df[temp_df['subject_id'].isin(value_counts[value_counts <= opt.max_appearances].index)].sort_values(['subject_id','admittime'])
    temp_df = temp_df.sort_values(['subject_id', 'admittime', 'seq_num'])
    print("Len of dataframe after filtering visits: "+str(len(temp_df)))

    temp_df = df[df['subject_id'].isin(temp_df['subject_id'])]
    value_counts = temp_df['icd_code'].value_counts()
    # Filter the DataFrame to keep only icd_code values that appear 5 or more times
    df = temp_df[temp_df['icd_code'].isin(value_counts[:opt.num_icd_code].index)].sort_values(['subject_id','admittime','seq_num'])

    print("Len of dataframe after filtering " + str(opt.num_icd_code) + " icd_code: "+str(len(df)))

    # Save icd_code to build tokenizer
    d_code = df[['icd_code', 'icd_version']].drop_duplicates()
    DXCCSR_v2023 = DXCCSR_v2023.rename(columns={"\'ICD-10-CM CODE\'": 'ICD-10-CM CODE',
                                                "\'ICD-10-CM CODE DESCRIPTION\'": 'ICD-10-CM CODE DESCRIPTION',
                                                "\'Default CCSR CATEGORY IP\'": 'Default CCSR CATEGORY IP',
                                                "\'Default CCSR CATEGORY DESCRIPTION IP\'": 'Default CCSR CATEGORY DESCRIPTION IP',
                                                "\'Default CCSR CATEGORY OP\'": 'Default CCSR CATEGORY OP',
                                                "\'Default CCSR CATEGORY DESCRIPTION OP\'": 'Default CCSR CATEGORY DESCRIPTION OP',
                                                "\'CCSR CATEGORY 1\'": 'CCSR CATEGORY 1',
                                                "\'CCSR CATEGORY 1 DESCRIPTION\'": 'CCSR CATEGORY 1 DESCRIPTION',
                                                "\'CCSR CATEGORY 2\'": 'CCSR CATEGORY 2',
                                                "\'CCSR CATEGORY 2 DESCRIPTION\'": 'CCSR CATEGORY 2 DESCRIPTION',
                                                "\'CCSR CATEGORY 3\'": 'CCSR CATEGORY 3',
                                                "\'CCSR CATEGORY 3 DESCRIPTION\'": 'CCSR CATEGORY 3 DESCRIPTION',
                                                "\'CCSR CATEGORY 4\'": 'CCSR CATEGORY 4',
                                                "\'CCSR CATEGORY 4 DESCRIPTION\'": 'CCSR CATEGORY 4 DESCRIPTION',
                                                "\'CCSR CATEGORY 5\'": 'CCSR CATEGORY 5',
                                                "\'CCSR CATEGORY 5 DESCRIPTION\'": 'CCSR CATEGORY 5 DESCRIPTION',
                                                "\'CCSR CATEGORY 6\'": 'CCSR CATEGORY 6',
                                                "\'CCSR CATEGORY 6 DESCRIPTION\'": 'CCSR CATEGORY 6 DESCRIPTION',})
    DXCCSR_v2023 = DXCCSR_v2023.applymap(lambda x: x.lstrip("\'") if isinstance(x, str) else x)
    DXCCSR_v2023 = DXCCSR_v2023.applymap(lambda x: x.rstrip("\'") if isinstance(x, str) else x)
    
    d_code = pd.merge(d_code, DXCCSR_v2023, right_on='ICD-10-CM CODE', left_on='icd_code', how='inner').sort_values(['CCSR CATEGORY 1', 
                                                                                                                            'CCSR CATEGORY 2',
                                                                                                                            'CCSR CATEGORY 3',
                                                                                                                            'CCSR CATEGORY 4',
                                                                                                                            'CCSR CATEGORY 5',
                                                                                                                            'CCSR CATEGORY 6'])
    d_code = d_code[['icd_code', 'icd_version']]
    d_code = pd.merge(d_code, d_icd_diagnoses, how='inner')
    d_code.to_csv(opt.d_code_dataset_path, index=False)

    # Build tokenizer
    tokenizer.build(path=opt.d_code_dataset_path, tokenizer_path=opt.tokenizer_path)
    tokenizer.auto_binary_encode()

    # Downcast
    for col in df.select_dtypes('number'):
        df[col] = pd.to_numeric(df[col], downcast='integer')
        if df[col].dtype == 'float':
            df[col] = pd.to_numeric(df[col], downcast='float')

    df = df.sort_values(['subject_id','admittime','seq_num'])
    df = df[[
            "subject_id",  
            'admittime',
            "seq_num", 
            'gender', 
            'anchor_age', 
            'anchor_year', 
            'icd_code',
            'icd_version', 
            'insurance', 
            'language', 
            'marital_status', 
            'race']]
    
    df['encode'] = df.swifter.apply(lambda x: tokenizer.encode(x['icd_code'], x['icd_version']), axis=1)
    insurance_dict = build_dict(df['insurance'])
    language_dict = build_dict(df['language'])
    marital_status_dict = build_dict(df['marital_status'])
    race_dict = build_dict(df['race'])
    gender_dict = build_dict(df['gender'])
    write_json(insurance_dict, opt.processing_directory, 'insurance_dict')
    write_json(language_dict, opt.processing_directory, 'language_dict')
    write_json(marital_status_dict, opt.processing_directory, 'marital_status_dict')
    write_json(race_dict, opt.processing_directory, 'race_dict')
    write_json(gender_dict, opt.processing_directory, 'gender_dict')
    df["insurance"] = df['insurance'].map(insurance_dict)
    df["language"] = df['language'].map(language_dict)
    df["marital_status"] = df['marital_status'].map(marital_status_dict)
    df["race"] = df['race'].map(race_dict)
    df["gender"] = df['gender'].map(gender_dict)
    print("==========================================Encoding done===========================================!")

    df = df.sort_values(['subject_id','admittime','seq_num'])
    df = df[["subject_id", 
             "anchor_age",
             "anchor_year", 
             "admittime", 
             "encode", 
             "gender", 
             "insurance", 
             "language", 
             "marital_status", 
             "race"]]
    
    grouped = df.groupby([
        "subject_id", 
        'anchor_age', 
        'anchor_year', 
        'admittime'], dropna=False)
    df = grouped.agg({
        "encode": lambda x: list(x),
        'gender': 'last', 
        "insurance": 'last',
        "language": 'last',
        "marital_status": 'last',
        "race": 'last',
    }).reset_index().sort_values(['subject_id','admittime'])

    df['year'] = pd.to_datetime(df['admittime']).dt.year
    df['month'] = pd.to_datetime(df['admittime']).dt.year
    df["age"] = df["year"] - df['anchor_year'] + df['anchor_age']
    df['admittime']=pd.to_datetime(df['admittime'])
    reverse_df = df[['subject_id', 'admittime']].groupby('subject_id')['admittime'].max().reset_index().rename(columns={'admittime': 'max_time'})
    df = pd.merge(df, reverse_df, how='inner')
    reverse_df = df[['subject_id', 'admittime']].groupby('subject_id')['admittime'].min().reset_index().rename(columns={'admittime': 'min_time'})
    df = pd.merge(df, reverse_df, how='inner')

    df['time'] = (df["admittime"] - df["min_time"]).dt.days
    df['reverse_time'] = (df["max_time"] - df['admittime']).dt.days

    df = df.sort_values(['subject_id','admittime'])
    df = df[["subject_id", 
             "age", 
             "time",
             "month", 
             "reverse_time", 
             "encode", 
             "gender", 
             "insurance", 
             "language", 
             "marital_status", 
             "race"]]
    grouped = df.groupby(["subject_id"], dropna=False)
    df = grouped.agg({
        "age": lambda x: list(x),
        "month": lambda x: list(x),
        "time": lambda x: list(x),
        "reverse_time": lambda x: list(x), 
        "encode": lambda x: list(x),
        'gender': lambda x: list(x),
        "insurance": lambda x: list(x),
        "language": lambda x: list(x),
        "marital_status": lambda x: list(x),
        "race": lambda x: list(x),
    }).reset_index().sort_values(['subject_id'])
    df['mask_attention'] = df['time'].apply(lambda x: len(x) * [True])

    validation_size = opt.validation_size
    test_size = opt.test_size
    # shuffle the DataFrame rows
    # df = df.sample(frac = 1)
    
    n = len(df)
    n_train = int(n*(1 - validation_size - test_size)/opt.batch_size) * opt.batch_size
    n_valid = int(n*validation_size/opt.batch_size) * opt.batch_size

    d_code = pd.read_csv(opt.d_code_dataset_path)
    category_size, category_type_size = ccsr_process(opt)
    print(df.columns)
    info_train = {
        "length": n_train,
        "attributes: ": list(df.columns),
        "num_label": len(d_code),
        "num_category": category_size,
        "num_category_type": category_type_size,
    }
    info_valid = {
        "length": n_valid,
        "attributes: ": list(df.columns),
        "num_label": len(d_code),
        "num_category": category_size,
        "num_category_type": category_type_size,
    }
    info_test = {
        "length": n - n_train - n_valid,
        "attributes: ": list(df.columns),
        "num_label": len(d_code),
        "num_category": category_size,
        "num_category_type": category_type_size,
    }
    info_dataset = {
        "attributes: ": list(df.columns),
        "num_label": len(d_code),
        "num_category": category_size,
        "num_category_type": category_type_size,
    }
    makedir(opt.dataset_directory, 'metadata')
    write_json(info_train, opt.metadata_dataset_directory, 'train')
    write_json(info_valid, opt.metadata_dataset_directory, 'valid')
    write_json(info_test, opt.metadata_dataset_directory, 'test')
    write_json(info_dataset, opt.metadata_dataset_directory, 'dataset')

    
    write_json(loads(df[:n_train].reset_index(drop=True).to_json(orient="columns")), opt.dataset_directory, 'train')
    write_json(loads(df[n_train:n_train+n_valid].reset_index(drop=True).to_json(orient="columns")), opt.dataset_directory, 'valid')
    write_json(loads(df[n_train+n_valid:].reset_index(drop=True).to_json(orient="columns")), opt.dataset_directory, 'test')

    
def build_dict(df):
    dict = {}
    for i in range(len(df)):
        if df.iloc[i] not in dict:
            dict[df.iloc[i]] = len(dict)
    for key, value in dict.items():
        sample = [0] * len(dict)
        sample[value] = 1
        dict[key] = sample 
    return dict


def get_weight(model):
    for name, param in model.named_parameters():
        print(str(name) + ": "+str(param))

def get_top_k(y_pred, top_k):
    topk, indices = torch.topk(y_pred, k=top_k)
    return torch.zeros(y_pred.shape, dtype=y_pred.dtype).scatter_(1, indices, topk).ceil()

# -------------
# -------------
# Load and save
# -------------
# -------------

def read_parquet(path):
    return pq.read_table(path).to_pandas()

def write_parquet(df, parient_dir, name):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, os.path.join(parient_dir, str(name) + '.parquet'))

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

def read_json(parient_dir, name=None):
    # Opening JSON file
    if name == None:
        path = parient_dir
    else:
        path = os.path.join(parient_dir, str(name) + ".json")
    with open(path, 'r') as openfile:
    
        # Reading from json file
        json_object = json.load(openfile)
    return json_object

def get_index(visits, index):
    return [item[:index] for item in visits], [item[index] for item in visits]

def load_model(path, model):
    path = os.path.join(path, f"model{MODEL_EXTENSION}")
    model.load_state_dict(torch.load(path))
    # return model

def save_model(path, model):
    path = os.path.join(path, f"model{MODEL_EXTENSION}")
    torch.save(model.state_dict(), path)

def save_report(path, report):
    with open(path, 'w') as file:
        # Write the report content to the file using write()
        file.write(report)

def save_checkpoint(checkpoint_directory, epoch, model, optimizer, LOSS, checkpoint_name=None):
    """
    The checkpoint will be saved in `checkpoint_directory` with name `checkpoint_name`.
    If `checkpoint_name` is None, the checkpoint will be saved with name `next_checkpoint_name_id + epoch`.
    """
    if checkpoint_directory is not None:
        if checkpoint_name is None:
            checkpoint_name = f'{epoch}{CHECKPOINT_EXTENSION}'

        path = os.path.join(checkpoint_directory, checkpoint_name)
        print("path: "+str(path))
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': LOSS,
                    }, path)

def load_checkpoint(PATH, model, optimizer):
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return epoch, model, optimizer, loss

def delete_folder(folder_path):
    try:
        # Use shutil.rmtree to delete the folder and its contents
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' successfully deleted.")
    except OSError as e:
        print(f"Error: {e.filename} - {e.strerror}")