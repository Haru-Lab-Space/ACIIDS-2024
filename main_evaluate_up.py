from torch.utils.data import DataLoader
import os
# =====================================================
from models.model import Model
from config.config import config
from train.Evaluator import Evaluator
from utils.utils import load_model, print_config, makedir, read_json
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
    print_config(opt, model)
    
    parient_dir = os.path.join(opt.model_directory, str(opt.batch_size))
    makedir(parient_dir, str(model.__class__.__name__))
    save_model_path = os.path.join(parient_dir, str(model.__class__.__name__))
    # Load model
    load_model(save_model_path, model)
    model.eval()

    test_data = CustomDataset(parient_dir=opt.dataset_directory,
                               keys=opt.keys, 
                               max_len=opt.max_appearances, 
                               split='test', 
                               tokenizer=tokenizer,
                               top_k_evaluate=opt.top_k,
                               opt=opt)

    evaluator = Evaluator(model, 
            tokenizer=tokenizer)
    
    evaluator.evaluate(test_data, 
              batch_size=opt.batch_size, 
              top_k=opt.top_k,
              max_len=opt.max_appearances,
              report_directory=opt.report_directory,
              predicted_directory=opt.predicted_directory,
              metric_name=opt.metric_name,
              shuffle=False,
              device=opt.device)
    