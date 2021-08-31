# Configuration file (powered by YACS).
# base on https://github.com/facebookresearch/pycls/blob/master/pycls/core/config.py
import argparse
import os
import sys
import time
from yacs.config import CfgNode


_C = CfgNode()
cfg = _C

# flag for one exp
_C.NAME = "debug"
_C.NUM_GPUS = [0]
# csv save name
_C.DESC = ""
_C.OUT_DIR = "./outputs/"
_C.PRETRAIN_WEIGHTS_PATH = "../../Datasets/pretrain/" # ./pretrain/ for cloud "../../Datasets/pretrain/" for local

# best or last
_C.TEST_CHECKPOINT = "best"
_C.TEST_SAVE = "/submission.csv"
_C.TEST_NFOLDS = True
_C.TTA = True
_C.FOLD = [0,1,2,3,4]

# Perform benchmarking to select fastest CUDNN algorithms (best for fixed input sizes)
_C.RNG_SEED = 1
_C.CUDNN_DETERMINISTIC = True
# set false for slower and repro
_C.CUDNN_BENCHMARK = True
_C.CUDNN_ENABLE = True
# sync_bn for multigpu
_C.SYNC_BN = True
_C.VAL_PERIOD = 20  # iterations

# ------- dataset -------- #
_C.DATA = CfgNode()
_C.DATA.TRAIN = CfgNode()
_C.DATA.TRAIN.TYPE = "cifar10"
_C.DATA.TRAIN.PATH = ""
 
_C.DATA.TRAIN.BATCH_SIZE = 16
_C.DATA.TRAIN.SHUFFLE = True
_C.DATA.TRAIN.SHUFFLE_PROB = 0.0 # for seti data
_C.DATA.TRAIN.NUM_WORKERS = 16
_C.DATA.TRAIN.IS_DISTRIBUTED = False
_C.DATA.TRAIN.USE_AUG = True
_C.DATA.TRAIN.USE_A_LIB = True
_C.DATA.TRAIN.DROPLAST = False
_C.DATA.TRAIN.AUG = [
    ["topil", []],
    ["horizontal_flip", [0.5]],
    ["rotation", [15]],
    ["random_resize_crop", [320]],
    # ["fix_resize", [int(224/0.875)]],
    # ["center_crop", [224]],
    # ["random_crop", [224, 20]],
    ["totensor", []],
    # ["normalization", [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]],
]

_C.DATA.VAL = CfgNode()
_C.DATA.VAL.TYPE = "cifar10"
_C.DATA.VAL.PATH = ""
_C.DATA.VAL.BATCH_SIZE = 16
_C.DATA.VAL.SHUFFLE = False
_C.DATA.VAL.SHUFFLE_PROB = 0.0
_C.DATA.VAL.IS_DISTRIBUTED = False
_C.DATA.VAL.DROPLAST = False
_C.DATA.VAL.NUM_WORKERS = 16
_C.DATA.VAL.USE_AUG = True
_C.DATA.VAL.USE_A_LIB = True
_C.DATA.VAL.AUG = [
    ["topil", []],
    ["fix_resize", [int(320/0.875)]],
    # ["center_crop", [320]],
    ["totensor", []],
    # ["normalization", None],
]

_C.DATA.TEST = CfgNode()
_C.DATA.TEST.TYPE = "cifar10"
_C.DATA.TEST.PATH = ""
_C.DATA.TEST.BATCH_SIZE = 16
_C.DATA.TEST.SHUFFLE_PROB = 0.0
_C.DATA.TEST.SHUFFLE = False
_C.DATA.TEST.IS_DISTRIBUTED = False
_C.DATA.TEST.DROPLAST = False
_C.DATA.TEST.NUM_WORKERS = 16
_C.DATA.TEST.USE_AUG = True
_C.DATA.TEST.USE_A_LIB = True
_C.DATA.TEST.AUG = [
    ["topil", []],
    ["fix_resize", [int(320/0.875)]],
    # ["center_crop", [320]],
    ["totensor", []],
    # ["normalization", None],
]

# ------ model ------- #
_C.MODEL = CfgNode()
_C.MODEL.NAME = "efficientnet_b3a"
_C.MODEL.PRETRAIN = True
_C.MODEL.SELF_HEAD = False
_C.MODEL.SELF_TAIL = False
_C.MODEL.OUTPUT_DIM = 10
_C.MODEL.NUM_CLASSES = [None, 512, 10] 
_C.MODEL.IN_CHANS = 3
_C.MODEL.DROPOUT = 0.5

# ------- Optimizer options --------- #
_C.OPTIM = CfgNode()
_C.OPTIM.OPT = "adamw"

# Learning rate ranges from BASE_LR to MIN_LR*BASE_LR according to the LR_POLICY
_C.OPTIM.LR = 1e-3
_C.OPTIM.MIN_LR = 1e-8
_C.OPTIM.WARMUP_LR = 1e-7
_C.OPTIM.MOMENTUM = 0.8
_C.OPTIM.OPT_BETAS = (0.9, 0.999)
_C.OPTIM.PCT_START = 0.1
_C.OPTIM.DIV_FACTOR_ONECOS = 1000
_C.OPTIM.FIN_DACTOR_ONCCOS = 1000
# L2 regularization
_C.OPTIM.WEIGHT_DECAY = 0.# 5e-4
_C.OPTIM.DECAY_RATE = 1                                                                                                                                 
_C.OPTIM.EPOCHS = 100
_C.OPTIM.WARMUP_EPOCHS = 0
_C.OPTIM.COOLDOWN_EPOCHS = 2
# Learning rate policy select from {'cos', 'exp', 'lin', 'steps', 'none'}
_C.OPTIM.LR_POLICY = "cosine"           

_C.TRAINOR = CfgNode()
_C.TRAINOR.NAME = "classification"
_C.TRAINOR.LOSSNAME = "bce_only_g"
_C.TRAINOR.LOG_PERIOD = 10  # interations
_C.TRAINOR.METRIC = "top1"
_C.TRAINOR.USE_MIXUP = False
_C.TRAINOR.ALPHA = 1.0
_C.TRAINOR.ALPHA_LOSS = 0.9
_C.TRAINOR.GAMMA = 1.0
_C.TRAINOR.SMOOTH_FACTOR = 0.1
_C.TRAINOR.GRAD_NORM_MAX = 0
_C.TRAINOR.APEX = False
# ---------------------------------- Default config ---------------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def dump_cfg(dump_path, cfg):
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(dump_path)
    with open(cfg_file, "w") as f:
        cfg.dump(stream=f)


def load_cfg(cfg_file):
    """Loads config from specified file."""
    with open(cfg_file, "r") as f:
        _C.merge_from_other_cfg(_C.load_cfg(f))


def reset_cfg():
    """Reset config to initial state."""
    _C.merge_from_other_cfg(_CFG_DEFAULT)


def load_cfg_fom_args(description="Config file options.", is_train=True):
    """Load config from command line arguments and set any specified options."""
    parser = argparse.ArgumentParser(description=description)
    help_s = "Config file location"
    parser.add_argument("--cfg", dest="cfg_file", help=help_s, required=True, type=str)
    help_s = "See pycls/core/config.py for all options"
    parser.add_argument("opts", help=help_s, default=None, nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    load_cfg(args.cfg_file)
    _C.merge_from_list(args.opts)
    if is_train:
        time_local = time.localtime()
        name_expend = "%02d%02d_%02d%02d%02d_"%(time_local[1], time_local[2],time_local[3], time_local[4], time_local[5])
        _C.NAME = name_expend +_C.NAME
    
    