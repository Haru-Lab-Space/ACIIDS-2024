import os
import numpy as np
# =====================================================
from models.model import Model
from train.LossFunction import LossFunction
from config.config import config
from train.Trainer import Trainer
from train.Optimizer import Optimizer
from utils.utils import print_config, makedir, save_model, read_json, delete_folder
from train.Dataset import CustomDataset
from train.Tokenizer import Tokenizer
# =====================================================

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

if __name__ == '__main__':
    opt = config()
    tokenizer = Tokenizer()
    tokenizer.build(path=opt.d_code_dataset_path, tokenizer_path=opt.tokenizer_path, category_map_path=opt.category_map_path, category_type_map_path=opt.category_type_map_path)
    tokenizer.auto_binary_encode()
    metatdata = read_json(parient_dir=opt.metadata_dataset_directory, name='dataset')
    opt.input_size = metatdata.get("num_label")
    opt.category_size = metatdata.get("num_category")
    opt.category_type_size = metatdata.get("num_category_type")

    model = Model(opt.model, opt)
    loss_fn = LossFunction(opt.lossfunction)
    optimizer = Optimizer(model, optimizer_name=opt.optimizer, learning_rate=opt.learning_rate)
    
    print_config(opt, model, loss_fn, optimizer)

    train_data = CustomDataset(parient_dir=opt.dataset_directory,
                               keys=opt.keys, 
                               max_len=opt.max_appearances, 
                               split='train', 
                               tokenizer=tokenizer,
                               opt=opt)
    valid_data = CustomDataset(parient_dir=opt.dataset_directory,
                               keys=opt.keys, 
                               max_len=opt.max_appearances, 
                               split='valid', 
                               tokenizer=tokenizer,
                               opt=opt)

    train = Trainer(model, 
            loss = loss_fn, 
            optimizer = optimizer)
    
    # Save path
    # save_model_path = os.path.join(os.path.join(os.path.join(os.path.join("models", str(opt.top_k)), str(opt.batch_size)), str(model.__class__.__name__)), str(model.model.base.__class__.__name__))
    # save_model_path = os.path.join(os.path.join(os.path.join("models", str(opt.top_k)), str(opt.batch_size)), str(model.__class__.__name__))

    

    train.fit(train_data, 
              valid_data, 
              batch_size=opt.batch_size, 
              epochs=opt.epochs,
              with_clip=opt.with_clip,
              patience=opt.patience,
              min_delta=opt.min_delta, 
            #   top_k=opt.top_k,
              max_len=opt.max_appearances,
              save_directory=opt.save_directory,
              checkpoints_directory=opt.checkpoints_directory,
              report_directory=opt.report_directory,
              loss_directory=opt.loss_directory,
            #   train_option=train_option,
              shuffle=True,
              device=opt.device,
              num_workers=opt.num_workers)
    
    parient_dir = os.path.join(opt.model_directory, str(opt.batch_size))
    makedir(parient_dir, str(model.__class__.__name__))
    save_model_path = os.path.join(parient_dir, str(model.__class__.__name__))
    save_model(save_model_path, model)
    delete_folder(os.path.join(opt.checkpoints_directory, str(opt.batch_size) + "/" + str(model.__class__.__name__)))