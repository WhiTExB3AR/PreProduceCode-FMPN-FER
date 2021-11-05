import torch
import os
from PIL import Image
import random
import numpy as np
import pickle
import torchvision.transforms as transforms

from .ckplus_res import CKPlusResDataset
from .mmi_res import MMIResDataset
from .affectnet import AffectNetDataset
from .base_dataset import BaseDataset
from .userdata import UserDataset


def create_dataloader(opt):
    data_loader = DataLoader()
    data_loader.initialize(opt)
    return data_loader

# Khi cho dữ liệu vào model để học thì thông thường sẽ cho dữ liệu theo từng batch một.
# DataLoader sẽ giúp chúng ta lấy dữ liệu theo từng batch, shuffle dữ liệu 
# cũng như load dữ liệu song song với nhiều multiprocessing workers.

class DataLoader:
    def name(self):
        return self.dataset.name() + "_Loader"

    def create_dataset(self):
        # specify which dataset to load here
        loaded_dataset = os.path.basename(self.opt.data_root.strip('/'))
        if 'CK' in loaded_dataset:
            dataset = CKPlusResDataset()
        elif 'MMI' in loaded_dataset:
            dataset = MMIResDataset()
        elif 'Affect' in loaded_dataset:
            dataset = AffectNetDataset()
        else:
            # dataset = BaseDataset()
            dataset = UserDataset()
        dataset.initialize(self.opt)
        return dataset

    def initialize(self, opt):
        self.opt = opt
        self.dataset = self.create_dataset()
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size = opt.batch_size,
            shuffle = not opt.serial_batches,
            num_workers = int(opt.n_threads))

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data