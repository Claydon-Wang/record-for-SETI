import logging

import numpy as np 
import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .cifar import CIFAR
from .seti import *

'''
TODO: 1. support autoaugmentation
json file for augmentation 
check autoaug codes and enable custom augmentation
'''

class LoaderFactory(object):
    """
    return dataloader from selected type and dataset
    get_loader:
        cfg: config for cfg.data.train, .val or .test
    """
    def __init__(self):
        self.loader_table = {
            "cifar10": CIFAR,
            "seti": SetiDataset,
            "semi_seti": SetiSemiDataset,
            "soft_semi_seti": SoftSetiSemiDataset,
            "full_setiA": FullSemiDatasetA,
            "full_setiB": FullSemiDatasetB,
            "full_setiC": FullSemiDatasetC,
            "oldplusnew_seti": OldPlusNewSetiDataset,
            "shuffle_seti" :SpatialShuffleDataset,
            "soft_new_old": SoftSemiPlusOldDataset,
        }
        
        self.aug_table = {
            "topil": transforms.ToPILImage(),  # h * w * c 
            "horizontal_flip": transforms.RandomHorizontalFlip,
            "vertical_flip": transforms.RandomVerticalFlip,
            "RandomAffine": transforms.RandomAffine,
            "rotation": transforms.RandomRotation,
            "random_resize_crop": transforms.RandomResizedCrop,
            "random_crop": transforms.RandomCrop,           
            "fix_resize": transforms.Resize,
            "center_crop": transforms.CenterCrop,
            "totensor": transforms.ToTensor(),  # c * h * w
            "normalization": transforms.Normalize,
            "AOneOf": A.OneOf,
            "AResize": A.Resize, # numpy h * w * c
            "AHorizontalFlip": A.HorizontalFlip, 
            "AVerticalFlip": A.VerticalFlip,
            "AShiftScaleRotate": A.ShiftScaleRotate,
            "ARandomResizedCrop": A.RandomResizedCrop,
            "ACutout": A.Cutout,
            "AToTensorV2": ToTensorV2(), # c*h*w
        }    
    
    def get_aug(self, cfg, use_aug=False):
        if not use_aug:
            return None
        aug_items = []
        # print(cfg.AUG)
        for aug_key, aug_value in cfg.AUG:
            _aug_class = self.aug_table.get(aug_key)
            if len(aug_value) is not 0:
                _aug_instance = _aug_class(*aug_value)
            else:
                _aug_instance = _aug_class
            aug_items.append(_aug_instance)
        if cfg.USE_A_LIB:
            augmentation = A.Compose(aug_items)
        else:
            augmentation = transforms.Compose(aug_items)
        return augmentation
        
    def get_loader(self, cfg, loader_type, fold=0, data=None):
        """
        loader_type: "train", "val", "test"
        """

        try: 
            _data_class = self.loader_table.get(cfg.TYPE)    
        except:
            logging.error(f"Dataset type {cfg.TYPE} is not exist")
        augmentation = self.get_aug(cfg, cfg.USE_AUG)
        data = _data_class(cfg,  dtype=loader_type, augmentation=augmentation)         
        data.next_fold(fold)
            
#         if cfg.IS_DISTRIBUTED:
#             sampler = torch.utils.data.distributed.DistributedSampler(data, shuffle=True)
#         else:
#             sampler = torch.utils.data.RandomSampler(data)
        data_loader = torch.utils.data.DataLoader(
                data, 
                # sampler=sampler,
                batch_size=cfg.BATCH_SIZE, 
                shuffle=cfg.SHUFFLE, 
                num_workers=cfg.NUM_WORKERS,
                drop_last=cfg.DROPLAST,)
        logging.info(f"Init {loader_type} dataloader success") 
        return data_loader, data
    
    @staticmethod
    def inverse_preprocess(image):
        import cv2
        image = image.numpy().transpose((1,2,0)) * 255
        image = image.astype(np.uint8)
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        return image

    def test_loader(self, cfg, loader_type):
        import matplotlib.pyplot as plt
        try: 
            _data_class = self.loader_table.get(cfg.TYPE)    
        except:
            logging.error(f"Dataset type {cfg.TYPE} is not exist")
        augmentation = self.get_aug(cfg, cfg.USE_AUG)
        
        _data = _data_class(cfg,  dtype=loader_type, augmentation=augmentation)
        data_loader = torch.utils.data.DataLoader(
                _data, 
                batch_size=4, 
                shuffle=cfg.SHUFFLE, 
                num_workers=cfg.NUM_WORKERS,
                drop_last=False,) 

        for images, labels in data_loader:
            fig = plt.figure()
            for j, image in enumerate(images):
                image = LoaderFactory.inverse_preprocess(image)
                a = fig.add_subplot(2,2,j+1)
                a.set_title("img_ori_tensor_0%d"%labels[j].numpy())
                plt.imshow(image)
            # plt.show()
            plt.savefig("./debug_data_shuffle_02.png")
            break
