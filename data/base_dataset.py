import torch
import os
import csv # for write filename img to csv file
from PIL import Image
import random
import numpy as np
import pickle
import torchvision.transforms as transforms



class BaseDataset(torch.utils.data.Dataset):
    """docstring for BaseDataset"""
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return os.path.basename(self.opt.data_root.strip('/')) # strip('/') xoá dấu '/' ở đầu và đuôi của data_root

    def initialize(self, opt):
        self.opt = opt

        self.imgs_dir = os.path.join(self.opt.data_root, self.opt.imgs_dir) # opt.data_root = datasets/CKPlus và opt.imgs_dir = imgs -> imgs_dir = datasets/CKPlus/imgs
        filename = self.opt.train_csv if self.opt.mode == "train" else self.opt.test_csv # lấy tên của file img trong file csv -> filename = test_ids_8.csv
        self.cur_fold = os.path.splitext(filename)[0].split('_')[-1] # lấy đuôi id của file csv, vd: test_ids_8.csv -> cur_fold = 8
        self.imgs_name_file = os.path.join(self.opt.data_root, filename) # lấy tất cả các tên file img trong record của file csv -> imgs_name_file = datasets/CKPlus/test_ids_8.csv
        self.imgs_path = self.make_dataset(self.imgs_dir, self.imgs_name_file) # xem cách hoạt động bên line 35 ckplus_res.py hoặc line 23 affecnet.py -> imgs_path = duyệt qua từng dòng ảnh trên file csv

    def make_dataset(self, imgs_dir, imgs_name_file):
        return None

    def load_dict(self, pkl_path):
        saved_dict = {}
        with open(pkl_path, 'rb') as f:
            saved_dict = pickle.load(f, encoding='latin1')
        return saved_dict

    def get_img_by_path(self, img_path):
        # print('****img_path before isfile in base_dataset = ', img_path)
        assert os.path.isfile(img_path), "Cannot find image file: %s" % img_path
        # print('****img_path after isfile in base_dataset = ', img_path)
        # print("...getting image by path in csv file...")
        img_type = 'L' if self.opt.img_nc == 1 else 'RGB'
        return Image.open(img_path).convert(img_type)

    def img_transform(self, img, use_data_augment = False, norm_tensor = False, lucky_dict = {}):
        if norm_tensor:
            img2tensor = transforms.Compose([ # Để thực hiện nhiều phép biến đổi trên dữ liệu đầu vào, transforms hỗ trợ hàm compose để gộp các transforms lại.
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Tham số đầu là mean, tham số sau là std => T.Normalize(mean, std)
            ])
        else:
            img2tensor = transforms.ToTensor() # Dữ liệu mình lấy được ở trên thì ảnh ở dạng PIL image, mình cần convert về dạng Torch tensor để cho Pytorch xử lý và tính toán.

        img = transforms.functional.resize(img, self.opt.load_size) # opt.load_size = 320
        # on-the-fly data augmentation
        if self.opt.mode == "train" and use_data_augment:
            # scale and crop 
            # lucky_num = random.randint(0, 4)
            lucky_num_crop = random.randint(0, 4) if not lucky_dict else lucky_dict['crop']
            img = transforms.functional.five_crop(img, self.opt.final_size)[lucky_num_crop]
            # Horizontally flip
            lucky_num_flip = random.randint(0, 1) if not lucky_dict else lucky_dict['flip']
            if lucky_num_flip:
                img = transforms.functional.hflip(img)
            # update seed dict if needed
            if not lucky_dict:
                lucky_dict.update({'crop': lucky_num_crop, 'flip': lucky_num_flip})
        else:
            img = transforms.functional.five_crop(img, self.opt.final_size)[-1]  # center crop # opt.final_size = 299

        # print(lucky_dict)
        return img2tensor(img)

    def __len__(self):
        return len(self.imgs_path)













