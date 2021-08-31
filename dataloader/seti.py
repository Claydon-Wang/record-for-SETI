import numpy as np 
import torch 
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SetiDataset(torch.utils.data.Dataset):
    """
    Dataset using 6 channels by stacking them along time-axis

    Attributes
    ----------
    paths : tp.Sequence[FilePath]
        Sequence of path to cadence snippet file
    labels : tp.Sequence[Label]
        Sequence of label for cadence snippet file
    transform: albumentations.Compose
        composed data augmentations for data
    """

    def __init__(self, cfg, dtype, augmentation=None):
        """Initialize"""
        self.path = cfg.PATH
        self.dtype = dtype
        if dtype is not "test":
            self.id_all = pd.read_csv(cfg.PATH + "train_labels.csv")
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1086)
            self.id_all["fold"] = -1
            for fold_id, (_, val_idx) in enumerate(skf.split(self.id_all["id"], self.id_all["target"])):
                self.id_all.loc[val_idx, "fold"] = fold_id
            self.val_fold_id = 0
        else: 
            self.id_all = pd.read_csv(cfg.PATH + "sample_submission.csv")
        self.path_label = self.get_path_label()
        self.paths = self.path_label["paths"]
        self.labels = self.path_label.get("labels", None)
        self.transform = augmentation
       
    def next_fold(self, fold):
        self.val_fold_id = fold
        # self.val_fold_id %= 5
        self.path_label = self.get_path_label()
        self.paths = self.path_label["paths"]
        self.labels = self.path_label.get("labels", None)
       
    def get_path_label(self):
        """Get file path and target info."""
        if self.dtype == "train":
            df = self.id_all[self.id_all["fold"] != self.val_fold_id]
            path_label = {
            "paths": [ self.path + "train/" f"{img_id[0]}/{img_id}.npy" for img_id in df["id"].values],
            "labels": df["target"].values.astype("f")}
        elif self.dtype == "val":
            df = self.id_all[self.id_all["fold"] == self.val_fold_id]
            path_label = {
            "paths": [ self.path + "train/" f"{img_id[0]}/{img_id}.npy" for img_id in df["id"].values],
            "labels": df["target"].values.astype("f")}
        else: # test
            df = self.id_all
            path_label = {
            "paths": [ self.path + "test/" f"{img_id[0]}/{img_id}.npy" for img_id in df["id"].values],}
        return path_label

    def __len__(self):
        """Return num of cadence snippets"""
        return len(self.paths)

    def __getitem__(self, index):
        """Return transformed image and label for given index."""
        path = self.paths[index]
        img = self._read_cadence_array(path)
        if self.transform is not None:
            img = self.transform(image = img)["image"]
        # print(img.shape)
        img = img.float()
        if self.labels is not None:
            label = self.labels[index]
            return img, label 
        return img, f"{self.paths[index][-19:-4]}"

    def _read_cadence_array(self, path):
        """Read cadence file and reshape"""
        img = np.load(path)[[0, 2, 4]]  # shape: (6, 273, 256)
        #img = img.transpose([1,2,0])
        img = np.vstack(img)  # shape: (1638, 256)
        img = img.transpose(1, 0)  # shape: (256, 1638)
        img = img.astype("f")[..., np.newaxis]  # shape: (256, 1638, 1)
        return img
    
    
class SetiSemiDataset(SetiDataset):
    """ for pseudo labeling """ 
    def __init__(self, cfg, dtype, augmentation=None):
        super(SetiSemiDataset, self).__init__(cfg, dtype, augmentation=None)
        if dtype is not "test":
            self.id_semi = pd.read_csv(cfg.PATH + "semi_labels.csv")
            semi_data = {
                "paths": [ self.path + "test/" f"{img_id[0]}/{img_id}.npy" for img_id in self.id_semi["id"].values],
                "labels":[ float(0) if float(img_target) < 0.5 else float(1) for img_target in  self.id_semi["target"].values.astype("f")]
            }
            self.paths += semi_data["paths"]
            #print(len(self.paths))
            self.labels = [*self.labels, *semi_data["labels"]]
            #print(len(self.labels))
            self.transform = augmentation


class SoftSetiSemiDataset(SetiDataset):
    """ for soft pseudo labeling """ 
    def __init__(self, cfg, dtype, augmentation=None):
        super(SoftSetiSemiDataset, self).__init__(cfg, dtype, augmentation=None)
        if dtype is not "test":
            self.id_semi = pd.read_csv(cfg.PATH + "semi_labels.csv")
            semi_data = {
                "paths": [ self.path + "test/" f"{img_id[0]}/{img_id}.npy" for img_id in self.id_semi["id"].values],
                "labels":[ float(img_target) for img_target in  self.id_semi["target"].values.astype("f")]
            }
            self.paths += semi_data["paths"]
            #print(len(self.paths))
            self.labels = [*self.labels, *semi_data["labels"]]
            #print(len(self.labels))
            self.transform = augmentation

            
class FullSemiDatasetA(SetiDataset):
    """ data shape 819*256*2 """
    def __init__(self, cfg, dtype, augmentation=None):
        super(FullSemiDatasetA, self).__init__(cfg, dtype, augmentation=None)
        self.transform = augmentation 
    
    def _read_cadence_array(self, path):
        """Read cadence file and reshape"""
        img = np.load(path)[[0, 2, 4]]  # shape: (6, 273, 256)
        img_noise = np.load(path)[[1, 3, 5]]
        img_noise = np.vstack(img_noise)
        img_noise = img_noise.astype("f")[..., np.newaxis]
        #img = img.transpose([1,2,0])
        img = np.vstack(img)  # shape: (1638, 256)
        img = img.astype("f")[..., np.newaxis]
        img = np.concatenate((img, img_noise),axis=2)
        #print(img.shape)
        img = img.transpose(1, 0, 2)  # shape: (256, 1638)
        # print(img.shape)
     # shape: (256, 1638, 1)
        return img
      
        
class FullSemiDatasetB(SetiDataset):
    """ data shape 256*273*6 """
    def __init__(self, cfg, dtype, augmentation=None):
        super(FullSemiDatasetB, self).__init__(cfg, dtype, augmentation=None)
        self.transform = augmentation 
    
    def _read_cadence_array(self, path):
        """Read cadence file and reshape"""
        img = np.load(path)[[0, 2, 4]]  # shape: (3, 273, 256)
        img_noise = np.load(path)[[1, 3, 5]]
        img = np.concatenate((img, img_noise),axis=0)
        img = img.astype("f")
        img = img.transpose(2, 1, 0)  # shape: (256, 273, 6)
        return img  

    
class FullSemiDatasetC(SetiDataset):
    """ data shape 256*273*3 """
    def __init__(self, cfg, dtype, augmentation=None):
        super(FullSemiDatasetC, self).__init__(cfg, dtype, augmentation=None)
        self.transform = augmentation 
    
    def _read_cadence_array(self, path):
        """Read cadence file and reshape"""
        img = np.load(path)[[0, 2, 4]]  # shape: (3, 273, 256)
        img = img.astype("f")
        img = img.transpose(2, 1, 0)  # shape: (256, 273, 3)
        # print(img.shape)
     # shape: (256, 1638, 1)
        return img 
    
    
class OldPlusNewSetiDataset(SetiDataset):
    """ data shape 256*273*3 """
    def __init__(self, cfg, dtype, augmentation=None):
        super(OldPlusNewSetiDataset, self).__init__(cfg, dtype, augmentation=None)
       
        self.id_old_train_1 = pd.read_csv(cfg.PATH + "old_leaky_data/train_labels_old.csv")
        self.id_old_train_1 = self.id_old_train_1[self.id_old_train_1["target"] == 1]
        self.id_old_train_1.iloc[:, 0] = [f"old_leaky_data/train_old/{item[0]}/{item}.npy" for item in self.id_old_train_1["id"]]

        self.id_old_test_1 = pd.read_csv(cfg.PATH + "old_leaky_data/test_labels_old.csv")
        self.id_old_test_1 = self.id_old_test_1[self.id_old_test_1["target"] == 1]
        self.id_old_test_1.iloc[:, 0] = [f"old_leaky_data/test_old/{item[0]}/{item}.npy" for item in self.id_old_test_1["id"]]

        self.id_all.iloc[:, 0] = [f"train/{item[0]}/{item}.npy" for item in self.id_all["id"]]
        #print(self.id_old_test_1)
        #print(self.id_old_train_1)
        #print(self.id_all)
        self.id_all = pd.concat([self.id_all, self.id_old_train_1, self.id_old_test_1, ], ignore_index=True)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1086)
        self.id_all["fold"] = -1
        
        #print(self.id_all)
        for fold_id, (_, val_idx) in enumerate(skf.split(self.id_all["id"], self.id_all["target"])):
            #print(val_idx)
            self.id_all.loc[val_idx, "fold"] = fold_id
            
        self.val_fold_id = 0
        self.path_label = self.get_path_label()
        self.paths = self.path_label["paths"]
        self.labels = self.path_label.get("labels", None)
        self.transform = augmentation
       
    def get_path_label(self):
        """Get file path and target info."""
        if self.dtype == "train":
            df = self.id_all[self.id_all["fold"] != self.val_fold_id]
            path_label = {
            "paths": [ self.path + img_id for img_id in df["id"].values],
            "labels": df["target"].values.astype("f")}
        elif self.dtype == "val":
            df = self.id_all[self.id_all["fold"] == self.val_fold_id]
            path_label = {
            "paths": [ self.path + img_id for img_id in df["id"].values],
            "labels": df["target"].values.astype("f")}
        else: # test
            df = self.id_all
            path_label = {
            "paths": [ self.path + "test/" f"{img_id[0]}/{img_id}.npy" for img_id in df["id"].values],}
        return path_label
    
    
class SoftSemiPlusOldDataset(OldPlusNewSetiDataset):
    """ for soft pseudo labeling """ 
    def __init__(self, cfg, dtype, augmentation=None):
        super(SoftSemiPlusOldDataset, self).__init__(cfg, dtype, augmentation=None)
        self.transform = augmentation
    
    def next_fold(self, fold):
        self.val_fold_id = fold
        # self.val_fold_id %= 5
        self.path_label = self.get_path_label()
        self.paths = self.path_label["paths"]
        self.labels = self.path_label.get("labels", None)
        self.id_semi = pd.read_csv(self.path + "semi_labels.csv")
        semi_data = {
                "paths": [ self.path + "test/" f"{img_id[0]}/{img_id}.npy" for img_id in self.id_semi["id"].values],
                "labels":[ float(img_target) for img_target in  self.id_semi["target"].values.astype("f")]
            }
        self.paths += semi_data["paths"]
        #print(len(self.paths))
        self.labels = [*self.labels, *semi_data["labels"]]
        

class SpatialShuffleDataset(SetiDataset):
    """ data shape 256*273*3 """
    def __init__(self, cfg, dtype, augmentation=None):
        super(SpatialShuffleDataset, self).__init__(cfg, dtype, augmentation=None)
        self.transform = augmentation
        self.shuffle_prob = cfg.SHUFFLE_PROB
    
    def _read_cadence_array(self, path):
        """Read cadence file and reshape"""
        img = np.load(path)[[0, 2, 4]]  # shape: (3, 273, 256)
        # print("before:", img.shape)
        if np.random.rand(1) < self.shuffle_prob:
            np.random.shuffle(img) # shuffle in 50% prob
            # print("after:", img.shape)
        img = np.vstack(img)
        img = img.transpose(1, 0)  # shape: (256, 1638)
        img = img.astype("f")[..., np.newaxis]  # shape: (256, 1638, 1)
        return img 