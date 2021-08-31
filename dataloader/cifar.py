import numpy as np 
import torch 


def unpickle(file):
    import pickle
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


class CIFAR(torch.utils.data.Dataset):
    def __init__(self, cfg, dtype="train", augmentation=None):
        
        self.data_full = unpickle(cfg.PATH[0])
        self.data = np.array(self.data_full[b"data"])
        self.label = np.array(self.data_full[b"labels"])
        for i in range(len(cfg.PATH)-1):
            self.data_full = unpickle(cfg.PATH[i+1])
            self.data = np.concatenate((self.data, np.array(self.data_full[b"data"])))
            self.label = np.concatenate((self.label, np.array(self.data_full[b"labels"])))
        self.dtype = dtype
        self.aug = augmentation
        
    def __getitem__(self,index):
        data_final = self.data[index].reshape(3,32,32).transpose(1,2,0)
        if self.aug is not None:
            data_final = self.aug(data_final)
        if self.dtype == "test":
            return data_final
        else:
            label_final = torch.from_numpy(np.array(self.label[index]))
            return data_final,label_final

    def __len__(self):
        return self.label.shape[0]