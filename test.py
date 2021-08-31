import logging
import sys
import warnings
import time
import torch
import numpy as np
import csv
from dataloader import datafactory
from optimizer import optim_factory, scheduler_factory
from trainor.trainor_factory import TrainorFacotry
from model import factory
import utils
from utils.tools import set_random_seed, load_checkpoints,set_cfg_and_logger
import ttach as tta


def oof_eval(save_dir, res_name, save_name):
    res_final = np.zeros(39996)
    res_id = []
    for i in range(5):
        with open(f"{save_dir}{res_name}{i}.csv", "r") as res_csv:
            readCSV = csv.reader(res_csv, delimiter=',')
            for j, row in enumerate(readCSV):
                if j == 0:
                    continue
                if i == 0:
                    res_id.append(row[0])
                res_final[j] += float(row[1]) 
    res_final/=5
    with open(f"{save_dir}{save_name}", "w+") as save_csv:
        writer = csv.writer(save_csv, delimiter=",")
        writer.writerow(["id", "target"])
        for i in range(39995):
            writer.writerow([res_id[i], res_final[i+1]])


def main(cfg):
    # init
    set_cfg_and_logger(cfg)
    set_random_seed(cfg)
    cfg.freeze()
    torch.hub.set_dir(cfg.PRETRAIN_WEIGHTS_PATH)
    loader_factory = datafactory.LoaderFactory()
    trainor_factory = TrainorFacotry()
    test_loader, _ = loader_factory.get_loader(cfg.DATA.TEST, "test")
    model = factory.BasicImageModel(cfg.MODEL)
    
    if cfg.TTA:
        transforms = tta.Compose([
            #tta.HorizontalFlip(),
            #tta.VerticalFlip(),
            # tta.Rotate90(angles=[0, 180]),
            # tta.Scale(scales=[1, 2, 4]),
            #tta.Multiply(factors=[0.9, 1, 1.1]),
            #tta.FiveCrops(832,832),
           # tta.Add(values=[10,20,10])
            #tta.Resize(sizes=[(832,832), (928,928)], original_size=(832,832))
        ])
        model = tta.ClassificationTTAWrapper(model, transforms, merge_mode='mean')
    model = torch.nn.DataParallel(model, cfg.NUM_GPUS).cuda() if torch.cuda.is_available() else model
    logging.info(model)
    trainor = trainor_factory.get_trainor(cfg.TRAINOR, save_model_name=[cfg.OUT_DIR+cfg.NAME+"/best.pth",cfg.OUT_DIR+cfg.NAME+"/last_0.pth"])
    
    if cfg.TEST_NFOLDS: # default is true
        for i in range(5):
            load_checkpoints(model, cfg.OUT_DIR+cfg.NAME+f"/{cfg.TEST_CHECKPOINT}_{i}_9.pth") # best or last
            trainor.inference(model, test_loader, cfg.OUT_DIR+cfg.NAME+f"/{cfg.TEST_CHECKPOINT}_{i}.csv")
        oof_eval(cfg.OUT_DIR+cfg.NAME, f"/{cfg.TEST_CHECKPOINT}_", cfg.TEST_SAVE)
    else:
        load_checkpoints(model, cfg.OUT_DIR+cfg.NAME+f"/{cfg.TEST_CHECKPOINT}.pth")    
        trainor.inference(model, test_loader, cfg.OUT_DIR+cfg.NAME+cfg.TEST_SAVE)
    
    


    
if __name__ == "__main__":
    from utils.config import cfg
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    utils.config.load_cfg_fom_args(is_train=False)
    main(cfg)
