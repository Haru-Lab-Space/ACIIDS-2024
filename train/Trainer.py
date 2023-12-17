import torch
import numpy as np
from torch import nn, autograd
from torch.optim import Optimizer, AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
from utils.cast_device import cast_to_device, cast_dict_to_device
from utils.utils import makedir, get_index, save_model, save_checkpoint, load_checkpoint, write_json
from torch.utils.data import Dataset, DataLoader
from train.EarlyStopper import EarlyStopper
import time
import copy
import sys

class Trainer:
    """
    Trainer Wrapper modules. Every models will be called inside here.

    Example:
    >>> from train import Trainer
    >>> from train.callbacks import EarlyStopping
    >>> from models import GPT2

    >>> model = GPT2()
    >>> trainer = Trainer(model)
    >>> trainer.fit(
    >>>     train_generator,
    >>>     valid_generator,
    >>>     batch_size=256,
    >>>     epochs=40,
    >>>     checkpoint_directory='/checkpoints',
    >>>     callbacks=[EarlyStopping()]
    >>> )

    Args:
        train_generator (torch.utils.data.DataLoader): Training data generator.
        valid_generator (torch.utils.data.DataLoader): Validation data generator.
        model (torch.nn.Module): Model to be trained.
        loss (torch.nn.Module): Loss function to be used.
        optimizer (torch.optim.Optimizer): Optimizer to be used.
        batch_size (int): Batch size.
        epochs (int): Epochs.
        checkpoint_directory (str): Directory to save model checkpoints.
        callbacks (list): List of callbacks to be used.
        time_limit (int): Limit time for training process. If None, it will be ignored.
        device (str): Device to be used. Default: 'cuda' if available, otherwise 'cpu'.

    Methods:
        fit: Fit model with given data generator.
        load_weights: Load model's weights from checkpoint.

    Keyword Args:
        learning_rate (float): Learning rate. Default: 1e-5.
    """

    def __init__(
        self,
        model: nn.Module,
        loss=None,
        optimizer: Optimizer = None,
        time_limit: int = None,
        **kwargs
    ):
        # --- Data inputs args ---
        # The train_generator and valid_generator will be set in fit method
        self.train_dataloader = None
        self.valid_dataloader = None

        # --- Model configs args ---
        # The model variable will be set in fit method
        self.model = model
        self.loss_function = loss
        self.optimizer = optimizer

        # --- Training args ---
        self.batch_size = None
        self.epochs = None
        self.top_k = None
        self.save_directory = None

        # --- Checkpoint args ---
        self.checkpoint_directory = None

        # Define histories which includes train_loss and valid_loss
        self.histories = {
            'train_loss': [],
            'valid_loss': []
        }

        # --- Utils options args ---
        # If the training time reach the time_limit, the training process will be stopped
        # By default, the time_limit is set to None, which unlimit the training time
        self.time_limit = time_limit
        self.training_start_time = 0

        # Define device
        self.device = None

        # Define best valid loss for tracking best model
        self.best_valid_loss = 1e6
        self.best_weights = self.model.state_dict()

        # Define some default kwargs
        self.next_checkpoint_name_id = 1
        self.kwargs = kwargs


    def __before_training(self):
        """
        (private) Before training phase.
        --------------------------------
        This function will be called before `__training()`.
        By default, this function will call `on_train_begin()` method from every callbacks.
        """
        

    def __before_epoch_training(self):
        """
        (private) Before epoch training phase.
        --------------------------------------
        This function will be called before `__training_step()`.
        By default, this function will call `on_epoch_begin()` method from every callbacks.
        """
        

    def __after_training(self):
        """
        (private) After training phase.
        -------------------------------
        This function will be called after `__training()`.
        By default, this function will call `on_train_end()` method from every callbacks.
        """

        

    def __after_epoch_training(self):
        """
        (private) After epoch training phase.
        -------------------------------------
        This function will be called after `__training_step()`.
        By default, this function will call `on_epoch_end()` method from every callbacks.
        """
        

    
    def __training(self):
        """
        (private) Training phase.
        -------------------------
        The steps are:
        1. Define scaler for loss scaling.
        2. Iterate over training data.
        3. Call `__before_epoch_training()`.
        4. Train model.
        5. Validate model.
        6. Save every checkpoint.
        7. Save best checkpoint.
        8. Call `__after_epoch_training()`.
        """
        
        # Define scaler for loss scaling
        scaler = GradScaler()

        # Setup training Parallel
        if self.device == 'cuda' and torch.cuda.device_count() > 1:
            print("Training Parallel!")
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)
        
        # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
        # Iterate over training data
        for epoch in range(self.epochs):
            # Break if time limit reached
            if self.time_limit is not None and (time.time() - self.training_start_time) > self.time_limit:
                break

            # Before training
            self.__before_epoch_training()

            # Training
            self.model.train(True)
            avg_train_loss = self.__training_step(epoch, scaler)

            # Validate
            if self.valid_dataloader is not None:
                self.model.eval()
                # print("start valid")
                avg_valid_loss = self.__validation_step(epoch)
            else:
                avg_valid_loss = 0
            print("Epoch: " + str(epoch) +" --- Valid loss: "+str(avg_valid_loss))
            # Save checkpoint
            # self.__save_checkpoint(epoch)
            # print("best loss: "+str(self.best_valid_loss) + " ------ loss: "+str(avg_valid_loss))
            # Save best model
            if self.valid_dataloader is not None and avg_valid_loss < self.best_valid_loss:
                self.best_valid_loss = avg_valid_loss
                self.best_weights = copy.deepcopy(self.model.state_dict())
                save_checkpoint(self.checkpoint_directory, epoch, self.model, self.optimizer, avg_valid_loss)

            # Save history
            self.histories['train_loss'].append(avg_train_loss)
            self.histories['valid_loss'].append(avg_valid_loss)

            # After training
            self.__after_epoch_training()
            if self.early_stopper.early_stop(avg_valid_loss):      
                print("Early stop at epoch: "+str(epoch) + " with valid loss: "+str(avg_valid_loss))       
                break
            # scheduler.step()


    def fit(
        self,
        train_generator: Dataset,
        valid_generator: Dataset = None,
        batch_size: int = None,
        epochs: int = None,
        with_clip: float = 0,
        patience: int = 10,
        min_delta: float = 0,
        # top_k: int = 5,
        save_directory: str = None,
        loss_directory: str = None,
        checkpoints_directory: str = None,
        report_directory: str = None,
        # train_option: str = None,
        shuffle: bool = True,
        max_len: str = None,
        device: str = None,
        **kwargs
    ):
        """
        Fit model with given data generator.
        -------------------------------------
        Args:
            train_generator (torch.utils.data.Dataset): Training data generator.
            valid_generator (torch.utils.data.Dataset): Validation data generator.
            batch_size (int): Batch size.
            epochs (int): Epochs.
            checkpoint_directory (str): Directory to save model checkpoints.
            callbacks (list): List of callbacks to be used.

        Keyword Args:
            next_checkpoint_name_id (int): Next checkpoint name id. Default: 1.
            middleware_preds (function): Preprocessing prediction function. Default: None.

        The steps are:
        1. Set arguments.
        2. Call `__before_training()`.
        3. Call `__training()`.
        4. Call `__after_training()`.
        """
        # Set arguments
        self.batch_size = batch_size
        self.epochs = epochs
        # self.train_option = train_option
        # self.top_k = top_k
        self.shuffle = shuffle
        self.max_len = max_len
        self.with_clip = with_clip
        self.patience = patience
        self.min_delta = min_delta
        self.early_stopper = EarlyStopper(patience=self.patience, min_delta=self.min_delta)
        self.device = device
        
        self.save_directory = save_directory
        self.loss_directory = loss_directory
        self.checkpoints_directory = checkpoints_directory
        self.report_directory = report_directory
        # self.checkpoint_directory = os.path.join(os.path.join(os.path.join(os.path.join(os.path.join(save_directory, "checkpoints"), str(top_k)), str(batch_size)), str(self.model.__class__.__name__)), str(self.model.model.base.__class__.__name__))
        # self.parent_dir = os.path.join(os.path.join(os.path.join(save_directory, "checkpoints"), str(top_k)), str(batch_size))
        self.parent_dir = os.path.join(self.checkpoints_directory, str(batch_size))
        makedir(self.parent_dir, str(self.model.__class__.__name__))
        self.checkpoint_directory = os.path.join(self.parent_dir, str(self.model.__class__.__name__))
        # self.parent_dir = os.path.join(os.path.join(os.path.join(save_directory, "loss"), str(top_k)), str(batch_size))
        self.parent_dir = os.path.join(loss_directory, str(batch_size))
        makedir(self.parent_dir, str(self.model.__class__.__name__))
        self.loss_directory = os.path.join(self.parent_dir, str(self.model.__class__.__name__))

        # Load dataloader
        self.train_dataloader = DataLoader(train_generator, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=kwargs.get('num_workers', 0), drop_last=True)
        self.valid_dataloader = DataLoader(valid_generator, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=kwargs.get('num_workers', 0), drop_last=True)

        # Checker
        assert self.epochs is not None, "Epochs must be defined."

        # Define some default kwargs
        self.next_checkpoint_name_id = kwargs.get('next_checkpoint_name_id', 1)

        # Before training
        self.__before_training()

        # Training
        self.training_start_time = time.time()
        self.__training()

        self.model.load_state_dict(self.best_weights)
        # After training

        self.__after_training()
        write_json(self.histories, self.loss_directory, 'histories_loss')
        
    def __training_step(self, epoch, scaler):
        """
        (private) Training step.
        ------------------------
        The steps are:
        1. Iterate over training data.
        2. Zero your gradients for every batch.
        3. Enable autocasting for forward pass.
        4. Perform backward pass and optimization using scaler.
        """
        logger_message = f'Training epoch {epoch}/{self.epochs}'
        avg_train_loss = 0

        progress_bar = tqdm(self.train_dataloader,
                            desc=logger_message, initial=0, dynamic_ncols=True)
        for batch, data in enumerate(progress_bar):
            # Destructuring data
            # Cast to device
            data = cast_dict_to_device(
                data, device=self.device
            )
            # Zero your gradients for every batch
            self.optimizer.zero_grad(set_to_none=True)

            # Enable autocasting for forward pass
            with autocast():
                preds_list, target_list = self.model(data)
                # preds_list = torch.softmax(preds_list, dim=1)
                loss = self.loss_function(preds_list, target_list)
                avg_train_loss += loss.item()
            
            # Perform backward pass and optimization using scaler
            scaler.scale(loss).backward()
            
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(self.optimizer)
            #Gradient Value Clipping
            if self.with_clip != 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.with_clip)


            scaler.step(self.optimizer)
            scaler.update()
            
            # Update progress bar
            progress_bar.set_description(
                f'{logger_message} | Loss: {loss.item():.4f}| avg_loss: {avg_train_loss / (batch+1):.4f}|')
            
        return avg_train_loss / len(self.train_dataloader)

    def __validation_step(self, epoch):
        """
        (private) Validation step.
        --------------------------
        The steps are:
        1. Iterate over validation data.
        2. Disable gradient calculation for foward pass.
        """
        
        logger_message = f'Validation epoch {epoch}/{self.epochs}'
        avg_valid_loss = 0
        progress_bar = tqdm(self.valid_dataloader,
                            desc=logger_message, initial=0, dynamic_ncols=True)
        with torch.no_grad():
            for _, data in enumerate(progress_bar):
                # Destructuring data
                data = cast_dict_to_device(
                    data, device=self.device
                )
                    
                preds_list, target_list = self.model(data)
                # preds_list = torch.softmax(preds_list, dim=1)
                loss = self.loss_function(preds_list, target_list)
                avg_valid_loss += loss.item()

            # Update progress bar
            progress_bar.set_description(
                f'{logger_message} | Loss: {loss.item():.4f}')
        return avg_valid_loss / len(self.valid_dataloader)