import os
import math
import random
import errno

from copy import copy
import os
import urllib.request
from tqdm import tqdm


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = self.ema * 0.99 + self.val * 0.01


def inter_and_union(pred, mask, num_class):
    pred = np.asarray(pred, dtype=np.uint8).copy()
    mask = np.asarray(mask, dtype=np.uint8).copy()

    # 255 -> 0
    pred += 1
    mask += 1
    pred = pred * (mask > 0)

    inter = pred * (pred == mask)
    (area_inter, _) = np.histogram(inter, bins=num_class, range=(1, num_class))
    (area_pred, _) = np.histogram(pred, bins=num_class, range=(1, num_class))
    (area_mask, _) = np.histogram(mask, bins=num_class, range=(1, num_class))
    area_union = area_pred + area_mask - area_inter

    return area_inter, area_union


def preprocess(image1, image2, mask, flip=False, scale=False, crop=False):
    if isinstance(image1, np.ndarray):
        image1 = Image.fromarray(image1)
    if isinstance(image2, np.ndarray):
        image2 = Image.fromarray(image2)
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)
    if flip:
        if random.random() < 0.5:
            image1 = image1.transpose(Image.FLIP_LEFT_RIGHT)
            image2 = image2.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.5:
            image1 = image1.transpose(Image.FLIP_TOP_BOTTOM)
            image2 = image2.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        if random.random() < 0.5:
            image1 = image1.transpose(Image.ROTATE_90)
            image2 = image2.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)

    if scale:
        w, h = image1.size
        rand_log_scale = math.log(scale[0], 2) + random.random() * (math.log(scale[1], 2) - math.log(scale[0], 2))
        random_scale = math.pow(2, rand_log_scale)
        new_size = (int(round(w * random_scale)), int(round(h * random_scale)))
        image1 = image1.resize(new_size, Image.ANTIALIAS)
        image2 = image2.resize(new_size, Image.ANTIALIAS)
        mask = mask.resize(new_size, Image.NEAREST)

    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    data_transforms = transforms.Compose(transform_list)

    image1 = data_transforms(image1)
    image2 = data_transforms(image2)
    mask = torch.LongTensor(np.array(mask).astype(np.int64))

    if crop:
        h, w = image1.shape[1], image1.shape[2]
        pad_tb = max(0, crop[0] - h)
        pad_lr = max(0, crop[1] - w)
        image1 = torch.nn.ZeroPad2d((0, pad_lr, 0, pad_tb))(image1)
        image2 = torch.nn.ZeroPad2d((0, pad_lr, 0, pad_tb))(image2)
        mask = torch.nn.ConstantPad2d((0, pad_lr, 0, pad_tb), 255)(mask)

        h, w = image1.shape[1], image1.shape[2]
        i = random.randint(0, h - crop[0])
        j = random.randint(0, w - crop[1])
        image1 = image1[:, i:i + crop[0], j:j + crop[1]]
        image2 = image2[:, i:i + crop[0], j:j + crop[1]]
        mask = mask[i:i + crop[0], j:j + crop[1]]

    return image1, image2, mask


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()


def adjust_learning_rate(optimizer, base_lr, max_iters, cur_iters, power=0.9):
    lr = base_lr * ((1 - float(cur_iters) / max_iters) ** power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


class CRF_Refiner(object):
    def __init__(self, shape):
        self.dcrf = __import__('pydensecrf.densecrf')

        self.d = self.dcrf.DenseCRF(shape[0], shape[1], 5)

    def __call__(self, softmax, image):
        """
        :param softmax: [C, H, W]
        :param image: [H, W, 3]
        :return:
        """
        # The input should be the negative of the logarithm of probability values
        # Look up the definition of the softmax_to_unary for more information
        unary = self.dcrf.utils.softmax_to_unary(softmax)

        # The inputs should be C-continious -- we are using Cython wrapper
        unary = np.ascontiguousarray(unary)
        self.d.setUnaryEnergy(unary)

        # This potential penalizes small pieces of segmentation that are
        # spatially isolated -- enforces more spatially consistent segmentations
        feats = self.dcrf.utils.create_pairwise_gaussian(sdims=(10, 10), shape=image.shape[:2])

        self.d.addPairwiseEnergy(feats, compat=3,
                                 kernel=self.dcrf.DIAG_KERNEL,
                                 normalization=self.dcrf.NORMALIZE_SYMMETRIC)

        # This creates the color-dependent features --
        # because the segmentation that we get from CNN are too coarse
        # and we can use local color features to refine them
        feats = self.dcrf.utils.create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20), img=image, chdim=2)

        self.d.addPairwiseEnergy(feats, compat=10,
                                 kernel=self.dcrf.DIAG_KERNEL,
                                 normalization=self.dcrf.NORMALIZE_SYMMETRIC)
        Q = self.d.inference(5)
        res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))
        return res


def safe_mkdir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise




DL_LINKS = {
    "table_1_plain": ["https://github.com/DIUx-xView/xView2_fifth_place/releases/download/final/weight.pth","dualhrnet_all_plain.pt"],
    "table_2_plain": ["https://data.goettingen-research-online.de/api/access/datafile/20228?gbrecs=true","dualhrnet_gupta_plain.pt"],
    "table_3_plain_1": ["https://data.goettingen-research-online.de/api/access/datafile/20230?gbrecs=true","dualhrnet_ood1_plain.pt"],
    "table_3_plain_2": ["https://data.goettingen-research-online.de/api/access/datafile/20234?gbrecs=true","dualhrnet_ood2_plain.pt"],
    "table_3_plain_3": ["https://data.goettingen-research-online.de/api/access/datafile/20238?gbrecs=true","dualhrnet_ood3_plain.pt"],
    "table_4_plain_g": ["https://data.goettingen-research-online.de/api/access/datafile/20228?gbrecs=true","dualhrnet_gupta_plain.pt"],
    "table_4_plain_1": ["https://data.goettingen-research-online.de/api/access/datafile/20230?gbrecs=true","dualhrnet_ood1_plain.pt"],
    "table_4_plain_2": ["https://data.goettingen-research-online.de/api/access/datafile/20234?gbrecs=true","dualhrnet_ood2_plain.pt"],
    "table_4_plain_3": ["https://data.goettingen-research-online.de/api/access/datafile/20238?gbrecs=true","dualhrnet_ood3_plain.pt"],
    "table_4_swa_g": ["https://data.goettingen-research-online.de/api/access/datafile/20229?gbrecs=true","dualhrnet_gupta_swa.pt"],
    "table_4_swa_1": ["https://data.goettingen-research-online.de/api/access/datafile/20231?gbrecs=true","dualhrnet_ood1_swa.pt"],
    "table_4_swa_2": ["https://data.goettingen-research-online.de/api/access/datafile/20235?gbrecs=true","dualhrnet_ood2_swa.pt"],
    "table_4_swa_3": ["https://data.goettingen-research-online.de/api/access/datafile/20239?gbrecs=true","dualhrnet_ood3_swa.pt"],
    "table_4_multiadabn_g": ["https://data.goettingen-research-online.de/api/access/datafile/20226?gbrecs=true","dualhrnet_gupta_multi.pt"],
    "table_4_multiadabn_1": ["https://data.goettingen-research-online.de/api/access/datafile/20258?gbrecs=true","dualhrnet_ood1_multi.pt"],
    "table_4_multiadabn_2": ["https://data.goettingen-research-online.de/api/access/datafile/20232?gbrecs=true","dualhrnet_ood2_multi.pt"],
    "table_4_multiadabn_3": ["https://data.goettingen-research-online.de/api/access/datafile/20236?gbrecs=true","dualhrnet_ood3_multi.pt"],
    "table_4_multiadabnswa_g": ["https://data.goettingen-research-online.de/api/access/datafile/20227?gbrecs=true","dualhrnet_gupta_multiswa.pt"],
    "table_4_multiadabnswa_1": ["https://data.goettingen-research-online.de/api/access/datafile/20257?gbrecs=true","dualhrnet_ood1_multiswa.pt"],
    "table_4_multiadabnswa_2": ["https://data.goettingen-research-online.de/api/access/datafile/20233?gbrecs=true","dualhrnet_ood2_multiswa.pt"],
    "table_4_multiadabnswa_3": ["https://data.goettingen-research-online.de/api/access/datafile/20237?gbrecs=true","dualhrnet_ood3_multiswa.pt"],
}

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_weights(setting_name):
    #TODO Put here the right setting -> downloadpath conversion
    dl_path, outfile = DL_LINKS[setting_name]
    filepath = "../weights/"+outfile
    print("Downloading from {} to {}".format(dl_path, filepath))
    if not os.path.isfile(filepath):
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=dl_path.split('/')[-1]) as t:
            urllib.request.urlretrieve(dl_path, filename = filepath, reporthook=t.update_to)
        print("Downloaded!")
    else:
        print("File existed allready!")
    return filepath

CONFIG_TREATER = {
    "table_4_multiadabnswa_1": "table_4_multi_1",
    "table_4_multiadabnswa_2": "table_4_multi_2",
    "table_4_multiadabnswa_3": "table_4_multi_3",
    "table_4_multiadabnswa_g": "table_4_multi_g",
    "table_4_multiadabn_1": "table_4_multi_1",
    "table_4_multiadabn_2": "table_4_multi_2",
    "table_4_multiadabn_3": "table_4_multi_3",
    "table_4_multiadabn_g": "table_4_multi_g",
    "table_4_swa_1": "table_4_plain_1",
    "table_4_swa_2": "table_4_plain_2",
    "table_4_swa_3": "table_4_plain_3",
    "table_4_swa_g": "table_4_plain_g",
}
