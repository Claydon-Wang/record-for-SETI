import logging
import sys
import warnings
import time
import torch
import csv
import os
import numpy as np

from optimizer import optim_factory, scheduler_factory
from trainor.trainor_factory import TrainorFacotry
from model import factory
from dataloader import datafactory

from utils.tools import set_cfg_and_logger, set_random_seed, load_checkpoints, cfg2csv
from utils.batchnorm import convert_model


def train_one_fold(cfg, loader_factory, trainor_factory, fold=0, previous_train_data=None, previous_val_data=None):
    
    train_loader, train_data = loader_factory.get_loader(cfg.DATA.TRAIN, "train", fold, previous_train_data)
    val_loader, val_data = loader_factory.get_loader(cfg.DATA.VAL, "val", fold, previous_val_data)
    test_loader, _ = loader_factory.get_loader(cfg.DATA.TEST, "test")
    
    # get model
    model = factory.BasicImageModel(cfg.MODEL)
    if cfg.DATA.TRAIN.IS_DISTRIBUTED:
        # TODO fix this part
        torch.distributed.init_process_group("nccl", rank=[0], world_size=2)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=cfg.NUM_GPUS, find_unused_parameters=True).cuda()
    else:
        model = convert_model(model) #syncbn
        model = torch.nn.DataParallel(model, cfg.NUM_GPUS).cuda() if torch.cuda.is_available() else model
    logging.info(model)
    
    # get optimizer
    opt = optim_factory.create_optimizer(cfg.OPTIM, model)
    sche, num_epochs = scheduler_factory.create_scheduler(cfg.OPTIM, opt, total_steps=cfg.OPTIM.EPOCHS*len(train_loader))
    trainor = trainor_factory.get_trainor(cfg.TRAINOR, save_model_name=[cfg.OUT_DIR+cfg.NAME+f"/best_{fold}.pth",cfg.OUT_DIR+cfg.NAME+f"/last_{fold}"]) # save last 0-9 
    for epoch in range(num_epochs):
        start_time = time.time()
        metric = trainor.train(model, epoch, train_loader, opt, sche)
        if epoch % cfg.VAL_PERIOD == 0:
            metric_val = trainor.val(model, epoch, val_loader)   
        
        log_info = []
        metric.update(metric_val)
        for key, value in metric.items():
            log_info.append(f"{key}{value}\t")
        end_time = time.time() - start_time
        log_info.append("Epoch Time:%.2fmin"%(end_time/60))
        logging.info("".join(log_info))
        # if epoch == 0: break
        
    list4print = cfg2csv(cfg)
    head4print = cfg2csv(cfg, True) if not os.path.exists(f"{cfg.OUT_DIR}{cfg.DESC}") else None
    with open(f"{cfg.OUT_DIR}{cfg.DESC}", "a") as f:
        writer = csv.writer(f)
        if head4print is not None: writer.writerow(["model", "val", "best", "best_epoch", *head4print[len(head4print)//2:]]) 
        writer.writerow([cfg.NAME, cfg.MODEL.NAME, metric["Val Prec@1:"], metric["BestV Prec@1:"], metric["Best Epoch:"], *list4print])
        
    torch.cuda.empty_cache()
    trainor.inference(model, test_loader, cfg.OUT_DIR+cfg.NAME+f"/{cfg.NAME}_{fold}.csv")
    load_checkpoints(model, cfg.OUT_DIR+cfg.NAME+f"/best_{fold}.pth")    
    trainor.inference(model, test_loader, cfg.OUT_DIR+cfg.NAME+f"/{cfg.NAME}_best_{fold}.csv")
    return train_data, val_data, float(metric["BestV Prec@1:"])


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
    with open(f"{save_dir}{save_name}.csv", "w+") as save_csv:
        writer = csv.writer(save_csv, delimiter=",")
        writer.writerow(["id", "target"])
        for i in range(39995):
            writer.writerow([res_id[i], res_final[i+1]])


def main(cfg):
    # init
    set_cfg_and_logger(cfg)
    set_random_seed(cfg)
    cfg.freeze()
    
    loader_factory = datafactory.LoaderFactory()
    trainor_factory = TrainorFacotry()
    # for cloud
    torch.hub.set_dir(cfg.PRETRAIN_WEIGHTS_PATH)
    # test_loader = loader_factory.test_loader(cfg.DATA.TRAIN, "train")
    
    avg_acc = []
    train_data, val_data = None, None
    for i in cfg.FOLD:# 5 folds 
        logging.info(f"Start training fold {i}")
        train_data, val_data, acc = train_one_fold(cfg, loader_factory, trainor_factory, i, train_data, val_data)
        avg_acc.append(acc)

    with open(f"{cfg.OUT_DIR}{cfg.DESC}", "a") as f:
        writer = csv.writer(f)
        writer.writerow([" ", " ", np.array(acc).mean(), np.array(acc).std(), cfg.NAME])
    if len(cfg.FOLD) == 5:
        oof_eval(cfg.OUT_DIR+cfg.NAME, f"/{cfg.NAME}_", f"/{cfg.NAME}")
        oof_eval(cfg.OUT_DIR+cfg.NAME, f"/{cfg.NAME}_best_", f"/{cfg.NAME}_best")
        

if __name__ == "__main__":
    from utils.config import cfg, load_cfg_fom_args
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    load_cfg_fom_args()
    main(cfg)