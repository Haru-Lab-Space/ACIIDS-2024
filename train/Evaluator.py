import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, ndcg_score
from utils.cast_device import cast_to_device, cast_dict_to_device
from utils.utils import get_top_k, makedir, save_report, write_json
import os
from typing import Dict
import numpy as np
import time

class Evaluator():
    def __init__(self, model, tokenizer, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.kwargs = kwargs

        self.testing_loader = None
        self.top_k = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def evaluate(
        self, 
        test_generator: Dataset,
        batch_size: int = None,
        top_k: int = 5,
        max_len: int = 64,
        report_directory: str = None,
        predicted_directory: str = None,
        # train_option: str = None,
        shuffle: bool = False,
        metric_name: list = [],
        device: str = None,
        **kwargs):

        # Set arguments
        self.batch_size = batch_size
        self.top_k = top_k
        self.max_len = max_len
        self.shuffle = shuffle
        self.device = device
        self.metric_name = metric_name
        self.report_directory = report_directory
        self.predicted_directory = predicted_directory
        self.model_name = str(self.model.__class__.__name__)
        
        
        intermediate_path = os.path.join(str(self.top_k), str(self.batch_size))
        self.parent_dir = os.path.join(self.report_directory, intermediate_path)
        makedir(self.parent_dir, str(self.model.__class__.__name__))
        self.model_report_directory = os.path.join(self.parent_dir, str(self.model.__class__.__name__))
        self.parent_dir = os.path.join(self.predicted_directory, intermediate_path)
        makedir(self.parent_dir, str(self.model.__class__.__name__))
        self.predicted_dataset_directory = os.path.join(self.parent_dir, str(self.model.__class__.__name__))
        
        # Load dataloader
        self.testing_loader = DataLoader(test_generator, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=kwargs.get('num_workers', 0), drop_last=False)
        

        # Evaluate and create report
        pred_list, target_list = self.__testing_step()
        report_dict = self.__metric_step(pred_list, target_list, metric_name)

        # # Saving report
        self.__save_metric_report(report_dict)
        
        # pred_list, true_list = self.__save_predicted(pred_list, target_list)
        # test_generator.set_predicted_info(pred_list, true_list)
        # test_generator.save_predicted_info(self.predicted_dataset_directory, "predicted_dataset")
        

    def __testing_step(self):
        """
        (private) Validation step.
        --------------------------
        The steps are:
        1. Iterate over validation data.
        2. Disable gradient calculation for foward pass.
        """
    
        # Map device
        if self.device == 'cuda' and torch.cuda.device_count() > 1:
            print("Training Parallel!")
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        
        logger_message = f'Evaluate and report '
        progress_bar = tqdm(self.testing_loader,
                            desc=logger_message, initial=0, dynamic_ncols=True)
        
        # Generate predictions on the testing data in batches
        predicted_labels = []
        true_labels = []
        with torch.no_grad():
            for batch, data in enumerate(progress_bar):
                data = cast_dict_to_device(
                    data, device=self.device
                )
                preds_list, target_list = self.model(data)
                preds_list = torch.softmax(preds_list, dim=-1)
                for preds, target in zip(preds_list.detach().cpu(), target_list.detach().cpu()):
                    min_k = min(len(target.nonzero()), self.top_k)
                    topksample, indices = torch.topk(preds, k=min_k)
                    preds = torch.zeros(preds.shape, dtype=preds.dtype).scatter_(0, indices, topksample).ceil()
                    predicted_labels.append(preds.numpy().astype(np.int8))
                    true_labels.append(target.numpy().astype(np.int8))
                progress_bar.update()

                
                # dict_ta = {}
                # dict_ta["preds_list"] = preds.detach().cpu().tolist()
                # dict_ta["target_list"] = target.detach().cpu().tolist()
                # write_json(dict_ta, "/data/thesis/save_directory/interpreting_directory/only_1.json")
            progress_bar.close()
        return predicted_labels, true_labels
    
    def __metric_step(self, pred_list, target_list, metric_name):
        dict = {}
        if 'classification_report' in metric_name:
            print("classification_report")
            dict['classification_report'] = classification_report(y_true=target_list, y_pred=pred_list, target_names=list(self.tokenizer.keys()), output_dict=True)
        if 'ndcg_score' in metric_name:
            print("ndcg_score")
            dict['ndcg_score'] = ndcg_score(y_true=target_list, y_score=pred_list)
        return dict
    
    def __save_metric_report(self, dict):
        for metric, report in dict.items():
            if isinstance(report, Dict):
                path = os.path.join(self.model_report_directory, f"{metric}.json")
                temp_report = {str(k): v for k, v in report.items()}
                write_json(temp_report, path)
            else:
                path = os.path.join(self.model_report_directory, f"{metric}.txt")
                save_report(path, str(report))
                
    def __save_predicted(self, predicted_labels, true_labels):
        pred_list = []
        true_list = []
        pred_len = (self.max_len - 1)
        start_interval = (pred_len - 1) * self.batch_size
        end_interval = pred_len * self.batch_size
        num_epochs = int(len(predicted_labels) / pred_len / self.batch_size)
        logger_message = f'Save predicted overall: '
        overall_progress_bar = tqdm(num_epochs,
                            desc=logger_message, initial=0, dynamic_ncols=True)
        print("len predicted_labels: "+str(len(predicted_labels)))
        print("num_epochs: "+str(num_epochs))
        for epoch_index in range(num_epochs):
            start = epoch_index * end_interval
            print("start: "+str(start))
            for interval in range(start_interval, end_interval):  
                print(predicted_labels[start + interval])
                print("predicted_labels: "+str(np.nonzero(predicted_labels[start + interval].tolist())))  
                print("true_labels: "+str(np.nonzero(true_labels[start + interval].tolist())))  
                time.sleep(10)
                pred_list.append(self.tokenizer.binary_decode(predicted_labels[start + interval].tolist(), top_k=self.top_k))
                true_list.append(self.tokenizer.binary_decode(true_labels[start + interval].tolist(), top_k=self.top_k))    
            overall_progress_bar.update()
        overall_progress_bar.close()
        return pred_list, true_list