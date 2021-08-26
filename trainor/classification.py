import os
import time
import logging
import numpy as np
import torch
import random
from tqdm import tqdm
import csv
from .loss_functions import LossFactory
from .train_tools import SelfData, CalculateAcc, save_checkpoints, ROCAUC
from torch.cuda.amp import autocast, GradScaler

class Classification(object):
    def __init__(self, cfg, **kwargs):
        self.log_fre = cfg.LOG_PERIOD
        self.best_model_name, self.last_model_name = kwargs["save_model_name"][0], kwargs["save_model_name"][1]
        self.use_mixup = cfg.USE_MIXUP
        self.mix_up_alpha = cfg.ALPHA
        self.loss_factory = LossFactory()
        self.loss_func = self.loss_factory.get_loss_func(cfg.LOSSNAME, cfg)
        self.loss_total = []
        self.loss_val_total = []
        self.grad_norm = cfg.GRAD_NORM_MAX
        self.metric = cfg.METRIC
        if cfg.METRIC == "auc":
            self.get_roc = ROCAUC()
            self.acc_train_class = SelfData()
            self.acc_val_class = SelfData()     
        else:
            self.acc_train_class = CalculateAcc()
            self.acc_val_class = CalculateAcc()
        self.acc_total = []
        self.acc_val_total = [] 
        self.best_metric = [0, 0]
        self.loss_train_class = SelfData()
        self.loss_val_class = SelfData()
        self.GradScaler = GradScaler() 
        
    def mixup_data(self, x, t, use_cuda=True):
        if self.mix_up_alpha > 0:
            lam = np.random.beta(self.mix_up_alpha, self.mix_up_alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        t_a, t_b = t, t[index]
        return mixed_x, t_a, t_b, lam
        
    def train(self, model, epoch, train_loader, opt, lr_scheduler):
        model.train()
        data_begin = time.time()
        its_num = len(train_loader)
        self.acc_train_class.reset()
        self.loss_train_class.reset()
        for its, (imgs, targets)in enumerate(train_loader):
#             if its <= 600:
#                 continue
            dtime = time.time()-data_begin
            imgs = imgs.cuda() if torch.cuda.is_available() else imgs
            targets = targets.cuda() if torch.cuda.is_available() else targets

            opt.zero_grad()
            with autocast():
                if self.use_mixup:
                    mixed_imgs, t_a, t_b, lam = self.mixup_data(imgs, targets)
                    outputs = model(mixed_imgs)
                    loss = lam * self.loss_func(outputs.squeeze(), t_a) + (1 - lam) * self.loss_func(outputs.squeeze(), t_b)
                else:
                    outputs = model(imgs)
                    loss = self.loss_func(outputs.squeeze(),targets)
            self.GradScaler.scale(loss).backward()
            # loss.backward()
            #if self.grad_norm != 0: torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_norm)
            self.GradScaler.step(opt)
            self.GradScaler.update()
            # opt.step()
            lr_scheduler.step()
           
            ntime = time.time()-(dtime+data_begin)
            data_begin = time.time()
            self.loss_train_class.add_value(loss.cpu())
            if self.metric == "auc":
                try:
                    get_roc_results = self.get_roc(outputs.squeeze(), targets)
                except:
                    get_roc_results = 0.50
                self.acc_train_class.add_value(get_roc_results)
                current_acc = self.acc_train_class.avg()
            else:
                self.acc_train_class.add_value(outputs.cpu(), targets.cpu())
                current_acc = self.acc_train_class.print_()
            
            if its % self.log_fre == 0:
                lr = opt.param_groups[0]['lr']
                mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0
                logging.info("[%d][%d/%d]\tLoss: %.5f\tLr: %.1e\tData: %dms\tNet: %dms\tMPerGPU:%.2fGb\tPrec@1: %.4f"%
                    (epoch, its, its_num, loss, lr, dtime*1000, ntime*1000, mem, current_acc))
#             if its == 1000:
#                 break
        self.acc_total.append(current_acc)
        self.loss_total.append(self.loss_train_class.avg())
        save_checkpoints(f"{self.last_model_name}_{epoch%10}.pth", model)
        return {
            "Train Prec@1:": "%.4f"%(current_acc),
            }

    def val(self, model, epoch, val_loader):
        model.eval()
        data_begin = time.time()
        its_num = len(val_loader)
        self.acc_val_class.reset()
        self.loss_val_class.reset()
        target_all = []
        outputs_all = []
        
        with torch.no_grad():
            for its, (imgs, targets)in enumerate(val_loader):
                
                dtime = time.time()-data_begin
                imgs = imgs.cuda() if torch.cuda.is_available() else imgs
                targets = targets.cuda() if torch.cuda.is_available() else targets

                outputs = model(imgs)
                
                outputs_all.append(outputs.sigmoid().to('cpu').numpy())
                target_all.append(targets.to('cpu').numpy())
                
                loss = self.loss_func(outputs.squeeze(),targets)
                ntime = time.time()-(dtime+data_begin)
                data_begin = time.time()
                self.loss_val_class.add_value(loss.cpu())
                if self.metric == "auc":
                    pass
                else:
                    self.acc_val_class.add_value(outputs.cpu(), targets.cpu())
                    current_acc = self.acc_val_class.print_()
                if its % self.log_fre == 0:
                    lr = 0
                    mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0
                    logging.info("[%d][%d/%d]\tLoss: %.5f\tLr: %.1e\tData: %dms\tNet: %dms\tMPerGPU:%.2fGb"%
                        (epoch, its, its_num, loss, lr, dtime*1000, ntime*1000, mem))
#                 if its == 21:
#                     break 
            
            get_roc_results = self.get_roc(np.concatenate(outputs_all), np.concatenate(target_all))        
            self.acc_val_class.add_value(get_roc_results)
            current_acc = self.acc_val_class.avg()
            
            self.acc_val_total.append(current_acc)
            self.loss_val_total.append(self.loss_val_class.avg())
        if self.best_metric[0] < current_acc:
            self.best_metric[0] = current_acc
            self.best_metric[1] = epoch
            save_checkpoints(self.best_model_name, model)
        return {
            "Val Prec@1:": "%.4f"%(current_acc),
            "BestV Prec@1:": "%.4f"%(self.best_metric[0]),
            "Best Epoch:" :self.best_metric[1],
            }
        
    def inference(self, model, test_loader, save_path):
        model.eval()
        with torch.no_grad():
            with open(save_path, 'w') as out_file:
                writer = csv.writer(out_file)
                writer.writerow(["id", "target"])
                for imgs, target_ids in tqdm(test_loader):
                    imgs = imgs.cuda() if torch.cuda.is_available() else imgs
                    outputs = model(imgs) # should be bs*n for binary
                    outputs = torch.sigmoid(outputs)
                    for output, t_id in zip(outputs, target_ids):
                        #binary_result = 0 if output <= 0.5 else 1
                        writer.writerow([f"{t_id}", output.data.cpu().numpy().item()])
                    
    def summarize(self):
        pass




class DSAN(Classification):
    """https://github.com/easezyc/deep-transfer-learning/tree/master/UDA/pytorch1.0/DSAN"""
    def __init__(self, cfg, **kwargs):
        super(DSAN, self).__init__(cfg, **kwargs)
        self.lmmd_loss = self.loss_factory.get_loss_func("lmmdloss")
    
    def train(self, model, epoch, train_loader, test_loader, opt, lr_scheduler, lr_step=False):
        model.train()
        data_begin = time.time()
        its_num = len(train_loader)
        self.acc_train_class.reset()
        self.loss_train_class.reset()
        iter_target = iter(test_loader)
        
        for its, (imgs, targets)in enumerate(train_loader):
            data_test, _ = iter_target.next()
            if its % len(test_loader) == (len(test_loader)-2):
                iter_target = iter(test_loader)
                
#             if its <= 40:
#                 break

            dtime = time.time()-data_begin
            imgs = imgs.cuda() if torch.cuda.is_available() else imgs
            targets = targets.cuda() if torch.cuda.is_available() else targets
            data_test = data_test.cuda() if torch.cuda.is_available() else data_test
            
            opt.zero_grad()
            if self.use_mixup:
                mixed_imgs, t_a, t_b, lam = self.mixup_data(imgs, targets)
                outputs = model(mixed_imgs)
                loss_cls = lam * self.loss_func(outputs.squeeze(), t_a) + (1 - lam) * self.loss_func(outputs.squeeze(), t_b) 
                _, source_features = model(imgs, return_features=True)     
            else:
                outputs, source_features = model(imgs, return_features=True)
                loss_cls = self.loss_func(outputs.squeeze(),targets)
                
            outputs_test, test_features = model(data_test, return_features=True)
            loss_lmmd = self.lmmd_loss.get_loss(source_features, test_features, targets, torch.nn.functional.softmax(outputs_test, dim=1))
            lambd = 2 / (1 + math.exp(-10 * (epoch) / 20)) - 1
            loss = loss_cls + 0.5 * lambd * loss_lmmd
            
            loss.backward()
            opt.step()
            if lr_step:
                lr_scheduler.step()
            else:
                lr_scheduler.step(its/its_num+epoch)
            ntime = time.time()-(dtime+data_begin)
            data_begin = time.time()
            self.loss_train_class.add_value(loss.cpu())
            if self.metric == "auc":
                try:
                    get_roc_results = self.get_roc(outputs.squeeze(), targets)
                except:
                    get_roc_results = 0.50
                self.acc_train_class.add_value(get_roc_results)
                current_acc = self.acc_train_class.avg()
            else:
                self.acc_train_class.add_value(outputs.cpu(), targets.cpu())
                current_acc = self.acc_train_class.print_()
            
            if its % self.log_fre == 0:
                lr = opt.param_groups[0]['lr']
                mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0
                logging.info("[%d][%d/%d]\tLoss: %.5f\tCls: %.5f\tLr: %.1e\tData: %dms\tNet: %dms\tMPerGPU:%.2fGb\tPrec@1: %.4f"%
                    (epoch, its, its_num, loss, loss_cls, lr, dtime*1000, ntime*1000, mem, current_acc))
#             if its == 21:
#                 break
        self.acc_total.append(current_acc)
        self.loss_total.append(self.loss_train_class.avg())
        save_checkpoints(f"{self.last_model_name}_{epoch%10}.pth", model)
        return {
            "Train Prec@1:": "%.4f"%(current_acc),
            }
