"""xBD Dataset in Torch
"""

import re
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageEnhance
import os
import scipy.io as sio
import torchvision.transforms as tr
import torch.functional as F
import random

holdout_train = ["guatemala-volcano","hurricane-florence","hurricane-harvey","hurricane-matthew","hurricane-michael","mexico-earthquake","midwest-flooding","palu-tsunami","santa-rosa-wildfire","socal-fire","lower-puna-volcano","moore-tornado","nepal-flooding","portugal-wildfire","tuscaloosa-tornado","woolsey-fire"]
holdout_test = ["joplin-tornado","pinery-bushfire","sunda-tsunami"]
holdout2_train = ["guatemala-volcano","hurricane-florence","hurricane-harvey","hurricane-matthew","hurricane-michael","mexico-earthquake","midwest-flooding","palu-tsunami","santa-rosa-wildfire","socal-fire","nepal-flooding","joplin-tornado","pinery-bushfire","sunda-tsunami","tuscaloosa-tornado","lower-puna-volcano","woolsey-fire"]
holdout2_test = ["moore-tornado","portugal-wildfire"]
holdout3_train = ["guatemala-volcano","hurricane-florence","hurricane-harvey","hurricane-matthew","hurricane-michael","mexico-earthquake","midwest-flooding","palu-tsunami","santa-rosa-wildfire","socal-fire","nepal-flooding","joplin-tornado","pinery-bushfire","portugal-wildfire","moore-tornado","sunda-tsunami"]
holdout3_test = ["tuscaloosa-tornado","lower-puna-volcano","woolsey-fire"]
gupta_train = ["guatemala-volcano","hurricane-michael","santa-rosa-wildfire","hurricane-florence","midwest-flooding","palu-tsunami","socal-fire","hurricane-harvey","mexico-earthquake","hurricane-matthew","nepal-flooding"]
gupta_test = ["tuscaloosa-tornado","lower-puna-volcano","woolsey-fire","joplin-tornado","pinery-bushfire","portugal-wildfire","moore-tornado","sunda-tsunami"]
big_train = ["guatemala-volcano","hurricane-michael","santa-rosa-wildfire","hurricane-florence","midwest-flooding","palu-tsunami","socal-fire","hurricane-harvey","mexico-earthquake","hurricane-matthew","nepal-flooding", "tuscaloosa-tornado","lower-puna-volcano","woolsey-fire","joplin-tornado","pinery-bushfire","portugal-wildfire","moore-tornado","sunda-tsunami"]
big_test = []
hold = ['hurricane-matthew', 'hurricane-michael', 'socal-fire', 'hurricane-florence', 'palu-tsunami', 'midwest-flooding', 'mexico-earthquake', 'hurricane-harvey', 'santa-rosa-wildfire', 'guatemala-volcano']

class DisastersDatasetUnet(Dataset):
    __file = []
    __pre = []
    __post = []
    __mask0 = []
    __mask1 = []
    __mask2 = []
    __mask3 = []
    __mask4 = []
    disasters_in_training = {
        "guatemala-volcano": True,
        "hurricane-florence": True,
        "hurricane-harvey": True,
        "hurricane-matthew": False,
        "hurricane-michael": True,
        "mexico-earthquake": True,
        "midwest-flooding": True,
        "palu-tsunami": True,
        "santa-rosa-wildfire": True,
        "socal-fire": True,
        "joplin-tornado": True,
        "lower-puna-volcano": True,
        "moore-tornado": True,
        "nepal-flooding": True,
        "pinery-bushfire": True,
        "portugal-wildfire": True,
        "sunda-tsunami": False,
        "tuscaloosa-tornado": True,
        "woolsey-fire": False
    }

    __train = True
    im_ht = 0
    im_wd = 0
    dataset_size = 0
    normalizer = tr.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    normalize = False
        
    def __init__(self, dataset_folders, train=True, im_size=[256,256], transform=tr.ToTensor(), normalize = False, flip = False, rotate = False, rotate10 = False, color = False, cut = False, disaster_list = None, SWA = False):
        """Initialize dataset

        Args:
            dataset_folders (list): List of Paths to dataset folders
            train (bool, optional): If disaster_list is not given, use predefined Train/Test split (split II). Defaults to True.
            im_size (list, optional): Input image size. Defaults to [256,256].
            transform (torch Transforms, optional): Torch transforms to apply onto images. Defaults to tr.ToTensor().
            normalize (bool, optional): Whether to normalize to imagenet values, this is deprecated!. Defaults to False.
            flip (bool, optional): If True randomly flips images. Defaults to False.
            rotate (bool, optional): If True randomly randomly makes true-grid transforms. Overwrites Flip. Defaults to False.
            rotate10 (bool, optional): If True, rotates by angles of 10 degrees and resizes. Defaults to False.
            color (bool, optional): If True uses color augmentations. Defaults to False.
            disaster_list (list, optional): List of disasters to include. Overwrites train if given, else use all disasters according to flag train. Defaults to None.
            SWA (bool, optional): For Stochastic weight averaging, outputs concated pre- and post-images in getitem. Defaults to False.
        """        
        self.__train = train
        self.__file = []
        self.__pre = []
        self.__post = []
        if self.__train:
            self.__mask0 = []
            self.__mask1 = []
            self.__mask2 = []
            self.__mask3 = []
            self.__mask4 = []
            
        self.im_ht = im_size[0]
        self.im_wd = im_size[1]
        self.transform = transform
        self.normalize = normalize
        self.flip = flip
        self.rotate = rotate
        self.rotate10 = rotate10
        self.color = color
        self.cut = cut
        self.SWA = SWA
        
        keywords=["mask0"]

        for folder in dataset_folders:
            for file in sorted(os.listdir(folder)):
                if file.endswith(".png"):
                    filename = os.path.splitext(file)[0]
                    filename_fragments = filename.split("_")
                    samekeywords = list(set(filename_fragments) & set(keywords))
                    if len(samekeywords) == len(keywords):

                        if ((disaster_list is None) and (self.__train and self.disasters_in_training[filename_fragments[0]]) or ((not self.__train) and (not self.disasters_in_training[filename_fragments[0]]))) or ((disaster_list is not None) and (filename_fragments[0] in disaster_list)):
                            # 1. read mask
                            file = file.replace("._","")
                            self.__mask0.append(folder + file)
                            self.__mask1.append(folder + file.replace("mask0","mask1"))
                            self.__mask2.append(folder + file.replace("mask0","mask2"))
                            self.__mask3.append(folder + file.replace("mask0","mask3"))
                            self.__mask4.append(folder + file.replace("mask0","mask4"))
                            # 2. read file name
                            self.__file.append(filename.replace("_mask0",""))
                            # 3. read pre image
                            self.__pre.append(folder + file.replace("_mask0","_pre_disaster"))
                            # 4. read post image
                            self.__post.append(folder + file.replace("_mask0","_post_disaster"))

        self.dataset_size = len(self.__file)
        self.files = self.__file


    def __getitem__(self, index):
        img_pre = Image.open(self.__pre[index])
        img_post = Image.open(self.__post[index])
        img_pre = img_pre.resize((self.im_ht, self.im_wd))
        img_post = img_post.resize((self.im_ht, self.im_wd))

        mask0 = Image.open(self.__mask0[index])
        mask1 = Image.open(self.__mask1[index])
        mask2 = Image.open(self.__mask2[index])
        mask3 = Image.open(self.__mask3[index])
        mask4 = Image.open(self.__mask4[index])

        mask0 = mask0.resize((self.im_ht, self.im_wd))
        mask1 = mask1.resize((self.im_ht, self.im_wd))
        mask2 = mask2.resize((self.im_ht, self.im_wd))
        mask3 = mask3.resize((self.im_ht, self.im_wd))
        mask4 = mask4.resize((self.im_ht, self.im_wd))
        
        if self.flip and not self.rotate:
            if random.random() < 0.5:
                op = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM][int(random.random()*1.9999)]
                img_pre = img_pre.transpose(op)
                img_post = img_post.transpose(op)
                mask0 = mask0.transpose(op)
                mask1 = mask1.transpose(op)
                mask2 = mask2.transpose(op)
                mask3 = mask3.transpose(op)
                mask4 = mask4.transpose(op)

        if self.rotate:
            opidx = int(random.random()*7.9999)
            op = [None, None, Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270, Image.TRANSPOSE][opidx]
            if op is not None:
                img_pre = img_pre.transpose(op)
                img_post = img_post.transpose(op)
                mask0 = mask0.transpose(op)
                mask1 = mask1.transpose(op)
                mask2 = mask2.transpose(op)
                mask3 = mask3.transpose(op)
                mask4 = mask4.transpose(op)

        if self.rotate10:
            if random.random() < 0.3:
                angle = random.gauss(0,5)
                img_pre = img_pre.rotate(angle)
                img_post = img_post.rotate(angle)
                mask0 = mask0.rotate(angle)
                mask1 = mask1.rotate(angle)
                mask2 = mask2.rotate(angle)
                mask3 = mask3.rotate(angle)
                mask4 = mask4.rotate(angle)
            

        if self.color > 0 :
            if random.random() < self.color*0.4:
                brightness_pre = random.gauss(1,0.06)
                brightness_post = random.gauss(1,0.06)
                img_pre = ImageEnhance.Brightness(img_pre).enhance(brightness_pre)
                img_post = ImageEnhance.Brightness(img_post).enhance(brightness_post)
            if random.random() < self.color*0.4:
                contrast_pre = random.gauss(1,0.06)
                contrast_post = random.gauss(1,0.06)
                img_pre = ImageEnhance.Contrast(img_pre).enhance(contrast_pre)
                img_post = ImageEnhance.Contrast(img_post).enhance(contrast_post)
            if random.random() < self.color*0.3:
                color_pre = random.gauss(1,0.03)
                color_post = random.gauss(1,0.03)
                img_pre = ImageEnhance.Color(img_pre).enhance(color_pre)
                img_post = ImageEnhance.Color(img_post).enhance(color_post)
            if random.random() < self.color*0.3:
                sharpness_pre = random.gauss(1,0.03)
                sharpness_post = random.gauss(1,0.03)
                img_pre = ImageEnhance.Sharpness(img_pre).enhance(sharpness_pre)
                img_post = ImageEnhance.Sharpness(img_post).enhance(sharpness_post)
            


        img_pre_tr = self.transform(img_pre)
        img_post_tr = self.transform(img_post)

        if self.normalize:
            img_pre_tr = self.normalizer(img_pre_tr)
            img_post_tr = self.normalizer(img_post_tr)
        

        mask0_tr = torch.tensor(np.array(mask0)).unsqueeze_(0)
        mask1_tr = torch.tensor(np.array(mask1)).unsqueeze_(0)
        mask2_tr = torch.tensor(np.array(mask2)).unsqueeze_(0)
        mask3_tr = torch.tensor(np.array(mask3)).unsqueeze_(0)
        mask4_tr = torch.tensor(np.array(mask4)).unsqueeze_(0)
        mask_input_tr = torch.cat((mask0_tr,mask1_tr,mask2_tr,mask3_tr,mask4_tr),0)

        if self.cut:
            i = random.randint(0, 511)
            j = random.randint(0, 511)
            img_pre_tr = img_pre_tr[:, i:i + 512, j:j + 512]
            img_post_tr = img_post_tr[:, i:i + 512, j:j + 512]
            mask_input_tr = mask_input_tr[:, i:i + 512, j:j + 512]

        if not self.SWA:
            return img_pre_tr,img_post_tr, mask_input_tr, index, self.__file[index]
        else:
            return torch.cat((img_pre_tr,img_post_tr),0)

    def getfilename(self, index):
        return self.__file[index]

    def __len__(self):
        return self.dataset_size
