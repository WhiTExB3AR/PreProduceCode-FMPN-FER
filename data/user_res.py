from .base_dataset import BaseDataset
import torch
import os
import csv # for write filename img to csv file
from PIL import Image
import random
import numpy as np
import pickle
import torchvision.transforms as transforms



class UserDataResDataset(BaseDataset):
    """docstring for UserDataResDataset"""
    def __init__(self):
        super(UserDataResDataset, self).__init__()
        
    def initialize(self, opt):
        super(UserDataResDataset, self).initialize(opt)

    def make_dataset(self, imgs_dir, imgs_name_file):
        # do not use offline data augmentation
        imgs = []
        assert os.path.isfile(imgs_name_file), "File '%s' does not exist." % imgs_name_file
        self.filename2csv(imgs_dir, imgs_name_file) # gọi hàm duyệt ảnh thành từng dòng row trong file csv
        with open(imgs_name_file, 'r') as f: # duyệt từng dòng ảnh trong file csv
            lines = f.readlines()
            imgs = [os.path.join(imgs_dir, line.strip()) for line in lines]
            imgs = sorted(imgs)
        return imgs

    def __getitem__(self, index):
        data_dict = {}

        img_path = self.imgs_path[index]
        data_dict['img_path'] = img_path

        # real_cls = self.get_cls_by_path(img_path)
        # data_dict['real_cls'] = real_cls

        lucky_dict = {}
        # [0, 1] color 
        img_tensor = self.img_transform(self.get_img_by_path(img_path), self.opt.use_data_augment, norm_tensor = False, lucky_dict = lucky_dict)
        data_dict['img_tensor'] = img_tensor

        # [0, 1] gray
        img_tensor_gray = self.img_transform(self.get_img_by_path(img_path).convert('L'), self.opt.use_data_augment, norm_tensor = False, lucky_dict = lucky_dict)
        data_dict['img_tensor_gray'] = img_tensor_gray

        # [0, 1] gray
        # img_res_tensor = self.img_transform(self.get_img_res_by_cls(real_cls).convert('L'), self.opt.use_data_augment, norm_tensor = False, lucky_dict = lucky_dict)
        # data_dict['img_res_tensor'] = img_res_tensor

        return data_dict

# ------- Start: B3AR config code: nhận input 1 ảnh bất kỳ -------     

    # Hàm duyệt file ảnh trong thư mục imgs của user để đưa vào file csv
    def filename2csv(self, imgs_dir, imgs_name_file):
        f = open(imgs_name_file,'w') # csv file
        w = csv.writer(f)
        for path, dirs, files in os.walk(imgs_dir): # imgs folder
            for filename in files:
                w.writerow([filename])
        return imgs_name_file

# ------- End: B3AR config code: nhận input 1 ảnh bất kỳ -------