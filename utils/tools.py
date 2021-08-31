import os
import random
import torch
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
from collections import OrderedDict

from .config import dump_cfg


def set_logger(log_path=None, log_filename='log'):
    """
    the max log file size is 10M, and only save the last 5 files.
    save the log in the path log_path and name is log_filename 
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    handlers = [logging.StreamHandler()]
    if log_path is not None:
        os.makedirs(log_path, exist_ok=True)
        handlers.append(
            RotatingFileHandler(os.path.join(log_path, log_filename), 
                                maxBytes=10 * 1024 * 1024, backupCount=5))
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s: %(message)s', 
                        handlers=handlers, datefmt='%m-%d %H:%M:%S')
    # this is a filter to specific logger name 
    # logging.getLogger("matplotlib").setLevel(logging.WARNING)


def set_cfg_and_logger(cfg):
    """
    set logging file and print cfg
    """
    cfg_name = cfg.OUT_DIR + cfg.NAME +"/"+ cfg.NAME +".yaml"
    if not os.path.exists(cfg.OUT_DIR + cfg.NAME):
        os.mkdir(cfg.OUT_DIR + cfg.NAME)
    if not os.path.exists(cfg_name):
        dump_cfg(cfg_name, cfg)
    else:
        s_add = 10
        logging.info(f"Already exist cfg, add {s_add} to ran_seed to continue training")
        cfg.RNG_SEED += s_add

    set_logger(cfg.OUT_DIR + cfg.NAME, f"{cfg.NAME}.log")
    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("CUDA version: {}".format(torch.version.cuda))
    logging.info("{} GPUs".format(torch.cuda.device_count()))
    logging.info(cfg)
    logging.info("Setting logging and config success")

    
def cfg2csv(cfg, get_head=False, list4print=[]):
    for k, v in cfg.items():
        if isinstance(cfg[k], dict):
            cfg2csv(cfg[k], get_head, list4print)
        else: list4print.append(k) if get_head else list4print.append(v)
    return list4print
            

def set_random_seed(cfg):
    os.environ['PYTHONHASHSEED'] = str(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)
    torch.cuda.manual_seed(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = cfg.CUDNN_BENCHMARK#cfg.CUDNN_DETERMINISTIC
    #torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK
    #torch.backends.cudnn.enabled = cfg.CUDNN_ENABLE
    logging.info(f"Random Seed: {cfg.RNG_SEED}")

    
def load_checkpoints(model, save_path, lrs=None, load_name='model'):
#     try:
#         states = torch.load(save_path)
#         model.load_state_dict(states['model'])
#     except:
#         states = torch.load(save_path)
#         states_new = OrderedDict()
#         for key, value in states["model"].items():
#             key_new = "module.backbone." + key[13:]
#             #print(key_new)
#             states_new[key_new] = value
#         model.load_state_dict(states_new)
#     try:
#         states = torch.load(save_path)
#         model.load_state_dict(states['model_state'])
#     except:
#         states_new = OrderedDict()
#         for key, value in states["model_state"].items():
#             key_new = "module." + key[7:]
#             #print(key_new)
#             states_new[key_new] = value
#         model.load_state_dict(states_new)     

#     if lrs is not None:
#         opt.load_state_dict(states['opt_state'])
#         current_epoch = states['epoch']
#         lrs.load_state_dict(states['lrs'])
#     logging.info(f'Loading self-pretrained checkpoints success for {load_name}')
    try:
        try:
            states = torch.load(save_path)
            model.load_state_dict(states['model_state'])
        except:
            states_new = OrderedDict()
            for key, value in states["model_state"].items():
                key_new = "module.model." + key[7:]
                #print(key_new)
                states_new[key_new] = value
            model.load_state_dict(states_new)     

        if lrs is not None:
            opt.load_state_dict(states['opt_state'])
            current_epoch = states['epoch']
            lrs.load_state_dict(states['lrs'])
        logging.info(f'Loading self-pretrained checkpoints success for {load_name}')
    except:
        current_epoch = 0
        logging.info(f"No self-pretrained checkpoints for {load_name}")