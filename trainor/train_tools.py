import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


class SelfData(object):
    def __init__(self):
        self.value = 0
        self.counter = 0 + 1e-8

    def add_value(self,add_value):
        self.counter += 1
        if isinstance(add_value, torch.Tensor):
            add_value = add_value.data.cpu().numpy()
        self.value += add_value

    def avg(self):
        return self.value/self.counter

    def reset(self):
        self.value = 0
        self.counter = 0 + 1e-8


class CalculateAcc(object):
    def __init__(self,topk=1):
        self.count_success_a = 0 + 1e-8
        self.count = 0+ 1e-8
        self.topk = topk
        
    def add_value(self,output,target):
        self.count += output.shape[0]
        _, preds = output.data.topk(self.topk,1,True,True)
        preds = preds.t()
        for pred in preds:
            self.count_success_a += pred.eq(target.data.view_as(pred)).sum().numpy()

    def print_(self):
        return (self.count_success_a/self.count)
    
    def reset(self, topk=1):
        self.count_success_a = 0 + 1e-8
        self.count = 0+ 1e-8
        self.topk = topk


def save_checkpoints(save_path, model, opt=None, epoch=None, lrs=None):
    if lrs is not None:
        states = { 
            'model_state': model.state_dict(),
            'epoch': epoch + 1,
            'opt_state': opt.state_dict(),
            'lrs':lrs.state_dict(),}
    elif opt is None:
        states = { 
            'model_state': model.state_dict(),}
    else:
        states = { 
            'model_state': model.state_dict(),
            'epoch': epoch + 1,
            'opt_state': opt.state_dict(),
            'lrs':lrs.state_dict(),}
    torch.save(states, save_path)
    

class ROCAUC(nn.Module):
    """ROC AUC score"""
    def __init__(self, average="macro") -> None:
        """Initialize."""
        self.average = average
        super(ROCAUC, self).__init__()

    def forward(self, y, t) -> float:
        """Forward."""
        if isinstance(y, torch.Tensor):
            y = y.cpu().detach().numpy()
        if isinstance(t, torch.Tensor):
            t = t.cpu().detach().numpy()

        return roc_auc_score(t, y, average=self.average)
