"""Utilities for handling data
"""
import json
import os
from os import path
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mpcol
import shapely.wkt
import shapely.geometry
from cv2 import fillPoly, imwrite
from PIL import Image

DMG_CLASSES = {"no-damage":0,"minor-damage":1,"major-damage":2,"destroyed":3}
DMG_NAMES = {0:"no-damage",1:"minor-damage",2:"major-damage",3:"destroyed"}

IMG_FOLDER = '../train/images/'
IMG_FILE = 'hurricane-harvey_00000114_post_disaster.png'
LABEL_FOLDER = '../train/labels/'
LABEL_FILE = 'hurricane-harvey_00000114_post_disaster.json'

def polygons_to_mask(PATH,out_type="one"):
    """
    :param PATH: A path to a JSON with Polygons
    :param out_type: changes output type
    :returns: If output type is one, returns a 1024x1024 np.array with values 1-4 corresponding to building dmg scales 0-4, where if polygons overlapped in the input, the maximum dmg was used. If output type is many returns a 1024x1024x4 np.array with values 0 or 1 corresponding to no building vs. building of dmg type x in channel x.
    """
    JSON = json.load(open(PATH))
    polygons = []
    for polygon in JSON['features']["xy"]:
        if (polygon['properties']['subtype'] != 'un-classified'):
            dmgtype = DMG_CLASSES[polygon['properties']['subtype']]
            coords = list(shapely.geometry.mapping(shapely.wkt.loads(polygon['wkt']))['coordinates'][0])
            polygons.append((dmgtype,np.array(coords, np.int32)))
    size = (1024,1024,5)
    mask_img = np.zeros(size, np.uint8)
    
    if out_type == "many":
        for poly in polygons:
            blank =  np.zeros((1024,1024), np.uint8)
            fillPoly(blank, [poly[1]], color=1)
            mask_img[:,:,poly[0]+1] = np.maximum(mask_img[:,:,poly[0]+1],blank)
        mask_img[:,:,0] = np.ones((1024,1024)) - np.maximum(np.maximum(np.maximum(mask_img[:,:,1],mask_img[:,:,2]),mask_img[:,:,3]),mask_img[:,:,4])
        return mask_img
    
    else:
        for poly in polygons:
            blank =  np.zeros((1024,1024), np.uint8)
            fillPoly(blank, [poly[1]], color=poly[0]+1)
            mask_img[:,:,poly[0]+1] = np.maximum(mask_img[:,:,poly[0]+1],blank)
        mask_all = np.maximum(np.maximum(np.maximum(mask_img[:,:,1],mask_img[:,:,2]),mask_img[:,:,3]),mask_img[:,:,4])
        return mask_all

def create_mask_png(IN_FOLDER,IN_FILE,OUT_FOLDER):
    """
    :param IN_FOLDER: A path to the input folder with jsons
    :param IN_FILE: name of input json
    :param OUT_FOLDER: Path to output folder for mask pngs
    """
    mask_all = polygons_to_mask(IN_FOLDER+IN_FILE)
    mask = Image.fromarray(mask_all)
    colorize_mask_(mask)
    mask.save(OUT_FOLDER+IN_FILE[:-19]+"_mask.png")

def create_multiple_mask_pngs(IN_FOLDER,IN_FILE,OUT_FOLDER):
    """
    :param IN_FOLDER: A path to the input folder with jsons
    :param IN_FILE: name of input json
    :param OUT_FOLDER: Path to output folder for mask pngs
    """
    mask_img = polygons_to_mask(IN_FOLDER+IN_FILE,out_type="many")
    Image.fromarray(mask_img[:,:,0]).save(OUT_FOLDER+IN_FILE[:-19]+"_mask0.png")
    Image.fromarray(mask_img[:,:,1]).save(OUT_FOLDER+IN_FILE[:-19]+"_mask1.png")
    Image.fromarray(mask_img[:,:,2]).save(OUT_FOLDER+IN_FILE[:-19]+"_mask2.png")
    Image.fromarray(mask_img[:,:,3]).save(OUT_FOLDER+IN_FILE[:-19]+"_mask3.png")
    Image.fromarray(mask_img[:,:,4]).save(OUT_FOLDER+IN_FILE[:-19]+"_mask4.png")

def colorize_mask_(mask, color_map=None):
    """
    Attaches a color palette to a PIL image. So long as the image is saved as a PNG, it will render visibly using the
    provided color map.
    :param mask: PIL image whose values are only 0 to 4 inclusive
    :param color_map: np.ndarray or list of 3-tuples with 5 rows
    :return:
    """
    color_map = color_map or np.array([(0, 0, 0),  # 0=background --> black
                                       (128, 255, 0),  # no damage (or just 'building' for localization) --> green
                                       (255, 255, 0),  # minor damage --> yellow
                                       (255, 128, 0),  # major damage --> orange
                                       (255, 0, 0),  # destroyed --> red
                                       ])
    assert color_map.shape == (5, 3)
    mask.putpalette(color_map.astype(np.uint8))
    return None

def load_mask_png(PATH):
    """
    :param PATH: Input mask png PATH
    :returns: Numpy array of mask
    """
    mask_all = np.array(Image.open(PATH))
    return mask_all


def make_output_directory(INPUT_PATH,OUTPUT_PATH):
    """
    :param INPUT_PATH: Input Path of all numpy arrays
    :param OUTPUT_PATH: Output Path where all the images get saved structured likes this:
        ├── images
        │   ├── test_damage_00000_image.png
        │   ├── test_damage_00001_image.png
        │   ├── test_localization_00000_image.png
        │   ├── test_localization_00001_image.png
        │   └── ...
        ├── predictions
        │   ├── test_damage_00000_prediction.png
        │   ├── test_damage_00001_prediction.png
        │   ├── test_localization_00000_prediction.png
        │   ├── test_localization_00001_prediction.png
        │   └── ...
        └── targets
            ├── test_damage_00000_target.png
            ├── test_damage_00001_target.png
            ├── test_localization_00000_target.png
            ├── test_localization_00001_target.png
            └── ...
    """
    if not os.path.exists(OUTPUT_PATH+"images/"):
        os.mkdir(OUTPUT_PATH+"images/")

    if not os.path.exists(OUTPUT_PATH+"predictions/"):
        os.mkdir(OUTPUT_PATH+"predictions/")

    if not os.path.exists(OUTPUT_PATH+"targets/"):
        os.mkdir(OUTPUT_PATH+"targets/")

    for file in os.listdir(INPUT_PATH):
        if file.endswith("-images.npy"):
            print(file)
            img = (np.load(INPUT_PATH+file)*256).astype(np.uint8)
            pred = (np.load(INPUT_PATH+file.replace("-images.npy","-outs.npy"))).astype(np.uint8)
            targ = (np.load(INPUT_PATH+file.replace("-images.npy","-masks.npy"))).astype(np.uint8)
            #pred_dmg = np.argmax(pred,axis=1)
            #targ_dmg = np.argmax(targ,axis=1)

            for i in range(img.shape[0]):
                imwrite(OUTPUT_PATH+"images/test_damage_"+(file.replace("-images.npy","00").replace("OutMasks-unetsmall-batch-",""))+str(i)+"_pre.png",np.moveaxis(img[i,:3,:,:],0, -1))
                imwrite(OUTPUT_PATH+"images/test_damage_"+file.replace("-images.npy","00").replace("OutMasks-unetsmall-batch-","")+str(i)+"_post.png",np.moveaxis(img[i,3:,:,:],0, -1))
                

                
                #pred_max = pred[i,0,:,:]
                #for i in range(1,5):
                #    pred_max, pred_dmg = np.maximum(pred_max,pred[i,i,:,:])
                #pred_dmg = np.maximum(np.maximum(np.maximum(1*pred[i,0,:,:],2*pred[i,1,:,:]),3*pred[i,2,:,:]),4*pred[i,3,:,:])
                imwrite(OUTPUT_PATH+"predictions/test_damage_"+file.replace("-images.npy","00").replace("OutMasks-unetsmall-batch-","")+str(i)+"_prediction.png",pred[i,:,:])
                pred[pred != 0] = 1
                imwrite(OUTPUT_PATH+"predictions/test_localization_"+file.replace("-images.npy","00").replace("OutMasks-unetsmall-batch-","")+str(i)+"_prediction.png",pred[i,:,:])
                
                #targ_dmg = np.maximum(np.maximum(np.maximum(1*targ[i,0,:,:],2*targ[i,1,:,:]),3*targ[i,2,:,:]),4*targ[i,3,:,:])
                imwrite(OUTPUT_PATH+"targets/test_damage_"+file.replace("-images.npy","00").replace("OutMasks-unetsmall-batch-","")+str(i)+"_target.png",targ[i,:,:])
                targ[targ != 0] = 1
                imwrite(OUTPUT_PATH+"targets/test_localization_"+file.replace("-images.npy","00").replace("OutMasks-unetsmall-batch-","")+str(i)+"_target.png",targ[i,:,:])




