import argparse
import math
import os
from utils.utils import makedir


def config():
    parser = argparse.ArgumentParser('Argument for training')
    # Setting training parameters
    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=3,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--path', type=str, default='/data/thesis',
                        help='Path of folder')
    parser.add_argument('--MIMIC', type=str, default='/data/MIMIC',
                        help='Path of MIMIC folder')
    parser.add_argument('--co_map', type=bool, default=False,
                        help='Path of MIMIC folder')
    parser.add_argument('--co_map_path', type=str, default=None,
                        help='Path of MIMIC folder')
    parser.add_argument('--category_map_state', type=str, default=False,
                        help='Path of MIMIC folder')
    parser.add_argument('--category_type_map_state', type=str, default=False,
                        help='Path of MIMIC folder')
    parser.add_argument('--category_map_path', type=str, default="/data/thesis/dataset/category_map_path.csv",
                        help='Path of MIMIC folder')
    parser.add_argument('--category_type_map_path', type=str, default="/data/thesis/dataset/category_type_map_path.csv",
                        help='Path of MIMIC folder')
    parser.add_argument('--alpha_path', type=str, default="/data/thesis/dataset/alpha.csv",
                        help='Path of MIMIC folder')
    parser.add_argument('--category_size', type=int, default=543,
                        help='Path of MIMIC folder')
    parser.add_argument('--category_type_size', type=int, default=543,
                        help='Path of MIMIC folder')
    
    # Preprocessing
    parser.add_argument('--num_sample', type=int, default=-1,
                        help='Path of MIMIC folder')
    parser.add_argument('--min_appearances', type=int, default=3,
                        help='Path of MIMIC folder')
    parser.add_argument('--max_appearances', type=int, default=64,
                        help='Path of MIMIC folder')
    parser.add_argument('--max_appearances_icd_code', type=int, default=5,
                        help='Path of MIMIC folder')
    parser.add_argument('--test_size', type=int, default=0.2,
                        help='Path of MIMIC folder')
    parser.add_argument('--keys', nargs="*",
                        type=str, default=[],
                        help='Path of MIMIC folder')


    # Training
    parser.add_argument('--model', type=str, default="RNN",
                        help='Name of model')
    parser.add_argument('--lossfunction', type=str, default="",
                        help='Name of loss function')
    parser.add_argument('--optimizer', type=str, default="",
                        help='Name of optimizer')
    parser.add_argument('--validation_size', type=float, default=0.2,
                        help='')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='path to pre-trained model')
    parser.add_argument('--patience', type=int, default=5,
                        help='path to pre-trained model')
    parser.add_argument('--min_delta', type=float, default=0,
                        help='path to pre-trained model')
    parser.add_argument('--head_training', type=str, default=None,
                        help='manner in which backbone was trained')
    
    # Evaluate
    parser.add_argument('--metric_name', nargs="*",
                        type=str, default=[],
                        help='Name of model')
    
    # Setup device
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--parallel', type=int, default=1, help='data parallel')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers to use')
    parser.add_argument('--pin_memory', type=bool, default=True,
                        help='')
    parser.add_argument('--shuffle', type=bool, default=False,
                        help='')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='100',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--with_clip', type=float, default=0,
                        help='with_clip')
                        
    # Setting task parameters
    parser.add_argument('--input_size', type=int, default=17877,
                        help='Number of labels')
    parser.add_argument('--num_ancestor', type=int, default=526,
                        help='Number of ancestors')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Predict top k code')
    parser.add_argument('--type_anchor_dataset', type=str, default="full_age",
                        help='Type of dataset')
    parser.add_argument('--type_dataset', type=str, default="encode",
                        help='Type of dataset')
    parser.add_argument('--size_dataset', type=str, default="mini_20000",
                        help='Size of dataset')

    # Option
    parser.add_argument('--have_ancestor', type=bool, default=False,
                        help='Number of ancestors')
    parser.add_argument('--have_time', type=bool, default=False,
                        help='Number of ancestors')

    
    # Basic parameters
    parser.add_argument('--drop_out', type=float, default=0.5,
                        help='Dimension of embedding') 
    
    # RNN
    parser.add_argument('--RNN_num_layers', type=int, default=1,
                        help='Number of layer in RNN') 
    parser.add_argument('--RNN_embedding_dim', type=int, default=256,
                        help='Dimension of hidden layer in RNN') 
    parser.add_argument('--RNN_hidden_size', type=int, default=64,
                        help='Dimension of hidden layer in RNN') 
    parser.add_argument('--RNN_dropout', type=int, default=0.5,
                        help='Dropout applied in RNN') 

    # HiTANet
    parser.add_argument('--HiTANet_hidden_size', type=int, default=256,
                        help='Number of filter') 
    parser.add_argument('--HiTANet_dim_feedforward', type=int, default=2048,
                        help='Number of filter') 
    parser.add_argument('--HiTANet_time_hidden_size', type=int, default=64,
                        help='Number of filter') 
    parser.add_argument('--HiTANet_key_query_hidden_size', type=int, default=64,
                        help='Number of filter') 
    parser.add_argument('--HiTANet_key_time_hidden_size', type=int, default=64,
                        help='Number of filter') 
    parser.add_argument('--HiTANet_dropout', type=int, default=0.5,
                        help='Number of filter') 
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='Number of filter') 
    
    # TCNN
    parser.add_argument('--TCNN_hidden_size', type=int, default=256,
                        help='Number of filter') 
    parser.add_argument('--TCNN_dim_feedforward', type=int, default=2048,
                        help='Number of filter') 
    parser.add_argument('--TCNN_time_hidden_size', type=int, default=64,
                        help='Number of filter') 
    parser.add_argument('--TCNN_key_query_hidden_size', type=int, default=64,
                        help='Number of filter') 
    parser.add_argument('--TCNN_key_time_hidden_size', type=int, default=64,
                        help='Number of filter') 
    parser.add_argument('--TCNN_kernel_size', type=int, default=3,
                        help='Number of filter') 
    parser.add_argument('--TCNN_dropout', type=int, default=0.5,
                        help='Number of filter') 

    # TCNNF
    parser.add_argument('--TCNNF_key_time_hidden_size', type=int, default=64,
                        help='Number of filter') 
    
    # * Transformer
    parser.add_argument('--enc_num_layer', default=6, type=int, 
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_num_layer', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=8192, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=2048, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    
    parser.add_argument('--trans_dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--position_embedding', default='sinusoidal', type=str,
                        help="Type of positional embedding")

    # GRAM
    parser.add_argument('--GRAM_hidden_size', default=128, type=int,
                        help="")
    parser.add_argument('--GRAM_hidden_size_graph', default=128, type=int,
                        help="")
    
    # KAME
    parser.add_argument('--KAME_hidden_size', default=128, type=int,
                        help="")
    parser.add_argument('--KAME_hidden_size_graph', default=128, type=int,
                        help="")

    opt = parser.parse_args()
    opt.save_directory = os.path.join(opt.path, "save_directory")
    opt.dataset_directory = os.path.join(opt.path, "dataset")
    opt.data_directory = os.path.join(opt.path, "preprocessing_data")
    opt.checkpoints_directory = os.path.join(opt.save_directory, "checkpoints")
    opt.model_directory = os.path.join(opt.save_directory, "models")
    opt.report_directory = os.path.join(opt.save_directory, 'reports')
    opt.loss_directory = os.path.join(opt.save_directory, "loss")
    opt.predicted_directory = os.path.join(opt.save_directory, "predicted_directory")
    opt.interpreting_directory = os.path.join(opt.save_directory, "interpreting_directory")
    opt.processing_directory = os.path.join(opt.path, "preprocessing_data")

    opt.metadata_dataset_directory = os.path.join(opt.dataset_directory, 'metadata')
    opt.d_code_dataset_path = os.path.join(opt.dataset_directory, 'd_code_dataset_path'+'.csv')
    opt.tokenizer_path = os.path.join(opt.dataset_directory, 'tokenizer'+'.json')

    makedir(opt.path, "save_directory")
    makedir(opt.path, "dataset")
    makedir(opt.path, "preprocessing_data")
    makedir(opt.save_directory, "checkpoints")
    makedir(opt.save_directory, "models")
    makedir(opt.save_directory, "reports")
    makedir(opt.save_directory, "loss")
    makedir(opt.save_directory, "predicted_directory")
    makedir(opt.save_directory, "interpreting_directory")
    # opt.encode_ancestorsListforeachLeaf_path = os.path.join(opt.data_directory, "graph/encode/ancestorsListforeachLeafFile.pk")
    # opt.encode_leavesListforeachAncestor_path = os.path.join(opt.data_directory, "graph/encode/leavesListforeachAncestorFile.pk")

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # warm-up for large-batch training,
    if opt.warm:
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate


    opt.hosp = os.path.join(opt.MIMIC, 'mimic-iv-2.2/hosp')
    opt.icu = os.path.join(opt.MIMIC, 'mimic-iv-2.2/icu')
    opt.admissions = os.path.join(opt.hosp, 'admissions'+'.csv')
    opt.d_hcpcs = os.path.join(opt.hosp, 'd_hcpcs'+'.csv')
    opt.d_icd_diagnoses = os.path.join(opt.hosp, 'd_icd_diagnoses'+'.csv')
    opt.d_icd_procedures = os.path.join(opt.hosp, 'd_icd_procedures'+'.csv')
    opt.d_labitems = os.path.join(opt.hosp, 'd_labitems'+'.csv')
    opt.df = os.path.join(opt.hosp, 'df'+'.csv')
    opt.diagnoses_icd = os.path.join(opt.hosp, 'diagnoses_icd'+'.csv')
    opt.drgcodes = os.path.join(opt.hosp, 'drgcodes'+'.csv')
    opt.emar_detail = os.path.join(opt.hosp, 'emar_detail'+'.csv')
    opt.emar = os.path.join(opt.hosp, 'emar'+'.csv')
    opt.hcpcsevents = os.path.join(opt.hosp, 'hcpcsevents'+'.csv')
    opt.labevents = os.path.join(opt.hosp, 'labevents'+'.csv')
    opt.microbiologyevents = os.path.join(opt.hosp, 'microbiologyevents'+'.csv')
    opt.omr = os.path.join(opt.hosp, 'omr'+'.csv')
    opt.patients_admission = os.path.join(opt.hosp, 'patients_admission'+'.csv')
    opt.patients = os.path.join(opt.hosp, 'patients'+'.csv')
    opt.pharmacy = os.path.join(opt.hosp, 'pharmacy'+'.csv')
    opt.poe_detail = os.path.join(opt.hosp, 'poe_detail'+'.csv')
    opt.poe = os.path.join(opt.hosp, 'poe'+'.csv')
    opt.prescriptions = os.path.join(opt.hosp, 'prescriptions'+'.csv')
    opt.procedures_icd = os.path.join(opt.hosp, 'procedures_icd'+'.csv')
    opt.provider = os.path.join(opt.hosp, 'provider'+'.csv')
    opt.services = os.path.join(opt.hosp, 'services'+'.csv')
    opt.transfers = os.path.join(opt.hosp, 'transfers'+'.csv')

    return opt


def preprocess_config():
    parser = argparse.ArgumentParser('Argument for training')
    parser.add_argument('--path', type=str, default='/data/thesis',
                        help='Path of folder')
    parser.add_argument('--MIMIC', type=str, default='/data/MIMIC',
                        help='Path of MIMIC folder')
    parser.add_argument('--CCSR', type=str, default='/data/CCSR',
                        help='Path of MIMIC folder')
    
    # Preprocessing
    parser.add_argument('--num_icd_code', type=int, default=256,
                        help='Path of MIMIC folder')
    parser.add_argument('--min_appearances', type=int, default=3,
                        help='Path of MIMIC folder')
    parser.add_argument('--max_appearances', type=int, default=64,
                        help='Path of MIMIC folder')
    parser.add_argument('--min_appearances_icd_code', type=int, default=5,
                        help='Path of MIMIC folder')
    parser.add_argument('--test_size', type=float, default=0.15,
                        help='Path of MIMIC folder')
    parser.add_argument('--validation_size', type=float, default=0.15,
                        help='Path of MIMIC folder')
    parser.add_argument('--batch_size', type=float, default=64,
                        help='Path of MIMIC folder')
    opt = parser.parse_args()
    
    opt.hosp = os.path.join(opt.MIMIC, 'mimic-iv-2.2/hosp')
    opt.icu = os.path.join(opt.MIMIC, 'mimic-iv-2.2/icu')
    opt.DXCCSR_v2023_path = os.path.join(opt.CCSR, 'DXCCSR_v2023-1.csv')
    opt.admissions = os.path.join(opt.hosp, 'admissions'+'.csv')
    opt.d_hcpcs = os.path.join(opt.hosp, 'd_hcpcs'+'.csv')
    opt.d_icd_diagnoses = os.path.join(opt.hosp, 'd_icd_diagnoses'+'.csv')
    opt.d_icd_procedures = os.path.join(opt.hosp, 'd_icd_procedures'+'.csv')
    opt.d_labitems = os.path.join(opt.hosp, 'd_labitems'+'.csv')
    opt.df = os.path.join(opt.hosp, 'df'+'.csv')
    opt.diagnoses_icd = os.path.join(opt.hosp, 'diagnoses_icd'+'.csv')
    opt.drgcodes = os.path.join(opt.hosp, 'drgcodes'+'.csv')
    opt.emar_detail = os.path.join(opt.hosp, 'emar_detail'+'.csv')
    opt.emar = os.path.join(opt.hosp, 'emar'+'.csv')
    opt.hcpcsevents = os.path.join(opt.hosp, 'hcpcsevents'+'.csv')
    opt.labevents = os.path.join(opt.hosp, 'labevents'+'.csv')
    opt.microbiologyevents = os.path.join(opt.hosp, 'microbiologyevents'+'.csv')
    opt.omr = os.path.join(opt.hosp, 'omr'+'.csv')
    opt.patients_admission = os.path.join(opt.hosp, 'patients_admission'+'.csv')
    opt.patients = os.path.join(opt.hosp, 'patients'+'.csv')
    opt.pharmacy = os.path.join(opt.hosp, 'pharmacy'+'.csv')
    opt.poe_detail = os.path.join(opt.hosp, 'poe_detail'+'.csv')
    opt.poe = os.path.join(opt.hosp, 'poe'+'.csv')
    opt.prescriptions = os.path.join(opt.hosp, 'prescriptions'+'.csv')
    opt.procedures_icd = os.path.join(opt.hosp, 'procedures_icd'+'.csv')
    opt.provider = os.path.join(opt.hosp, 'provider'+'.csv')
    opt.services = os.path.join(opt.hosp, 'services'+'.csv')
    opt.transfers = os.path.join(opt.hosp, 'transfers'+'.csv')

    
    opt.processing_directory = os.path.join(opt.path, "preprocessing_data")
    opt.feature_selection_path = os.path.join(opt.processing_directory, "feature_selection_path.csv")

    opt.dataset_directory = os.path.join(opt.path, "dataset")
    opt.metadata_dataset_directory = os.path.join(opt.dataset_directory, 'metadata')
    opt.d_code_dataset_path = os.path.join(opt.dataset_directory, 'd_code_dataset_path'+'.csv')
    opt.ccsr_category_path = os.path.join(opt.dataset_directory, 'ccsr_category_path'+'.csv')
    opt.category_map_path = os.path.join(opt.dataset_directory, 'category_map_path'+'.csv')
    opt.category_type_map_path = os.path.join(opt.dataset_directory, 'category_type_map_path'+'.csv')
    opt.tokenizer_path = os.path.join(opt.dataset_directory, 'tokenizer'+'.json')
    


    return opt