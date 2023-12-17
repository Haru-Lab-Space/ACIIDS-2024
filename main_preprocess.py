# =====================================================
from config.config import preprocess_config
from utils.utils import all_processing,  makedir
from train.Tokenizer import Tokenizer
# =====================================================

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

if __name__ == '__main__':
    opt = preprocess_config()
    makedir(opt.path, "preprocessing_data")
    makedir(opt.path, "dataset")
    # feature_selection_processing(opt)
    tokenizer = Tokenizer()
    all_processing(tokenizer, opt)
    # split_data_processing(tokenizer, opt)
    # systhetic_processing(tokenizer, opt)