"""Script for generating fused predictions.
Contains legacy code.

Example:
    From inside ddd folder:
    $ python3 utils/output_fusion.py "loc_folder" "dmg_folder" "out_folder" medianfreq
"""
import torch
import torch.nn as nn
import skimage.measure as measure
from scipy import interpolate
import numpy as np
from PIL import Image
import argparse
import shutil
import os
if __name__ == "__main__":
    from load import colorize_mask_
else:
    from utils.load import colorize_mask_
from tqdm import tqdm

def copy_twomodel_preds(loc_folder, dmg_folder, output_folder, mode = "interpolate"):
    """Merge predictions from two models

    Args:
        loc_folder (str): Path to localization predictions
        dmg_folder (str): Path to damage predictions
        output_folder (str): Output path
        mode (str, optional): One of copy, interpolate, meanfreq, maxfreq, medianfreq, weightedfreq. Defaults to "interpolate".
    """    
    if not (os.path.exists(loc_folder) and os.path.exists(dmg_folder)):
        raise ValueError("Paths do not exist")
    locpreds = sorted([pred for pred in os.listdir(os.path.join(loc_folder, "predictions")) if pred.startswith("test_localization_")])
    dmgpreds = sorted([pred for pred in os.listdir(os.path.join(dmg_folder, "predictions")) if pred.startswith("test_damage_")])

    if len(locpreds) != len(dmgpreds):
        raise ValueError("Folders do not contain the same amount of predictions... locpreds: {}, dmgpreds: {}".format(len(locpreds),len(dmgpreds)))

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    pred_folder = os.path.join(output_folder,"predictions")
    if not os.path.exists(pred_folder):
        os.mkdir(pred_folder)

    for locpred, dmgpred in tqdm(zip(locpreds, dmgpreds),total = len(locpreds)):
        if locpred.split("_")[-2] != dmgpred.split("_")[-2]:
            raise ValueError("Predictions {} and {} dont match!".format(locpred, dmgpred))
        
        locpath = os.path.join(loc_folder, "predictions", locpred)
        dmgpath = os.path.join(dmg_folder, "predictions", dmgpred)

        shutil.copy(locpath, pred_folder)
        if mode == "copy":
            shutil.copy(dmgpath, pred_folder)
        else:
            dmg = np.array(Image.open(dmgpath)).astype(float)
            v = np.unique(dmg)
            if 0 not in v:
                dmg_interpolated = dmg
            elif len(v) == 2:
                dmg[dmg == 0] = v[1]
                dmg_interpolated = dmg
            elif len(v) == 1:
                dmg[dmg == 0] = 1
                dmg_interpolated = dmg
            else:
                dmg[dmg == 0] = np.nan
                x = np.arange(0, dmg.shape[1])
                y = np.arange(0, dmg.shape[0])
                #mask invalid values
                dmg = np.ma.masked_invalid(dmg)
                xx, yy = np.meshgrid(x, y)
                #get only the valid values
                x1 = xx[~dmg.mask]
                y1 = yy[~dmg.mask]
                dmg_new = dmg[~dmg.mask]

                dmg_interpolated = interpolate.griddata((x1, y1), dmg_new.ravel(), (xx, yy), method='nearest')
            dmg = dmg_interpolated.astype(np.uint8)
            

            if mode in ["meanfreq","maxfreq","medianfreq","weightedfreq"]:
                loc = np.array(Image.open(locpath))
                labels = measure.label(loc,connectivity=1)
                num_labels = labels.max()
                empty = np.zeros(dmg.shape)
                for labeli in range(1,int(num_labels)):
                    dmg_region = np.where(labels == labeli, dmg, np.zeros(dmg.shape))
                    if mode == "meanfreq":
                        value = round(dmg_region[dmg_region!=0].mean())
                    elif mode == "maxfreq":
                        unique_values, unique_counts = np.unique(dmg_region[dmg_region!=0], return_counts = True)
                        value = unique_values[np.argmax(unique_counts)]
                    elif mode == "medianfreq":
                        value = np.median(dmg_region[dmg_region!=0])
                    elif mode == "weightedfreq":
                        unique_values, unique_counts = np.unique(dmg_region[dmg_region!=0], return_counts = True)
                        new_counts = unique_counts.copy()
                        if 2 in unique_values:
                            idx = np.where(unique_values == 2)[0]
                            count2 = unique_counts[idx]
                            if count2/sum(unique_counts) >= 0.1:
                                new_counts[idx] = 8 * count2
                        if 3 in unique_values:
                            idx = np.where(unique_values == 3)[0]
                            count3 = unique_counts[idx]
                            if count3/sum(unique_counts) >= 0.1:
                                new_counts[idx] = 4 * count3
                        if 4 in unique_values:
                            idx = np.where(unique_values == 4)[0]
                            count4 = unique_counts[idx]
                            if count4/sum(unique_counts) >= 0.1:
                                new_counts[idx] = 2 * count4
                        idx = np.where(new_counts.cumsum() >= int(sum(new_counts)/2))[0][0]
                        value = unique_values[idx]
                    empty += np.where(dmg_region == 0, dmg_region, value)
                dmg = empty.astype(np.uint8)

            dmg_img = Image.fromarray(dmg)
            colorize_mask_(dmg_img)
            dmg_img.save(os.path.join(pred_folder,dmgpred))


    shutil.copytree(os.path.join(loc_folder,"images"),os.path.join(output_folder,"images"))
    shutil.copytree(os.path.join(loc_folder,"targets"),os.path.join(output_folder,"targets"))

        #loc = np.array(Image.open(locpath))
        #dmg = np.array(Image.open(dmgpath))






def max_freq_per_component_fusion(loc_batch, dmg_batch):
    """LEGACY
    """    
    #print(loc_batch.shape)
    dmg_map = torch.argmax(dmg_batch,1)
    empty_map = torch.zeros_like(dmg_map)
    for i in range(loc_batch.shape[0]):
        #print(measure.label)
        labels = measure.label(loc_batch[i,:,:,:].squeeze().cpu().data,connectivity=1)
        labeled_loc = torch.Tensor(labels).to(0)
        num_labels = torch.max(labeled_loc)
        #print(num_labels)
        for labeli in range(1,num_labels.long()):
            dmg_region = torch.where(labeled_loc == labeli, dmg_map[i,:,:], torch.zeros_like(dmg_map[i,:,:]))
            unique_values, unique_counts = torch.unique(dmg_region[dmg_region!=0], return_counts = True)
            max_freq = unique_values[torch.argmax(unique_counts)]
            #print(max_freq)
            empty_map[i,:,:] += torch.where(dmg_region == 0, dmg_region, max_freq)
            
    return empty_map

class DeepMerge(nn.Module):
    """LEGACY
    Deep Merge for two stage architecture

    Usage is as follows: Train loc and dmg models seperately. Then feed the pretrained models in this init. Then set the optimizer to just use the parameters of DeepMerge.out (thus freezing the two models) and then train for a few epochs (should not be too long).

    """
    def __init__(
        self,
        loc_model,
        dmg_model,
        n_classes
    ):

        super(DeepMerge, self).__init__()

        self.loc_model = loc_model
        self.dmg_model = dmg_model

        self.out = nn.Conv2d(96, n_classes, kernel_size=1, stride=1)

    def forward(self, x):
        _, loc_features = self.loc_model(x, return_features = "dec1")
        _, dmg_features = self.dmg_model(x, return_features = "last")

        x = self.out(torch.cat((loc_features,dmg_features),1))

        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate smaller Img+Mask png dataset')
    parser.add_argument('loc_folder', type=str, metavar='str', help='loc folder to copy')
    parser.add_argument('dmg_folder', type=str, metavar='str', help='dmg folder to copy')
    parser.add_argument('out_folder', type=str, metavar='str', help='destination folder')
    parser.add_argument('--mode', type =str, default = 'interpolate', help = 'mode, one of copy, interpolate, meanfreq, maxfreq, medianfreq, weightedfreq')

    args = parser.parse_args()

    copy_twomodel_preds(args.loc_folder,args.dmg_folder,args.out_folder, mode = args.mode)