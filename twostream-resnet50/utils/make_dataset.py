"""Script for generating dataset

Example:
    Inside ddd folder for creation of 256x256 cuts
    $ python3 utils/make_dataset.py --data-folder datasets/train/ --output-folder datasets/resize256/ --mode manymulti --cutwidth 256 --resizewidth 256
    Inside ddd folder for creation of original 1024x1024 files
    $ python3 utils/make_dataset.py --data-folder datasets/hold/ --output-folder datasets/hold/all/ --mode nocut
"""
import load
import augment
import os
import re
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import time



def process_one(file, keywords, LABEL_FOLDER, IMG_FOLDER, MASK_FOLDER, args):
    if file.endswith(".json"):
        filename = os.path.splitext(file)[0]
        filename_fragments = filename.split("_")
        samekeywords = list(set(filename_fragments) & set(keywords)) #um nur die post jsons auszuwählen
        if len(samekeywords) == len(keywords):
            #print(LABEL_FOLDER)
            #print(file)
            load.create_multiple_mask_pngs(LABEL_FOLDER,file,MASK_FOLDER)
            augment.cut_and_resize_one(IMG_FOLDER,file[:-19]+"_pre_disaster.png",args.output_folder,args.cutwidth,args.resizewidth)
            augment.cut_and_resize_one(IMG_FOLDER,file[:-5]+".png",args.output_folder,args.cutwidth,args.resizewidth)
            for i in range(5):
                augment.cut_and_resize_one(MASK_FOLDER,file[:-19]+"_mask%s.png" % i,args.output_folder,args.cutwidth,args.resizewidth)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate smaller Img+Mask png dataset')
    parser.add_argument('--data-folder', type=str, default='../../xBD/train/', metavar='str', help='folder that contains data (default train)')
    parser.add_argument('--output-folder', type=str, default='../../xBD/resize/', metavar='str', help='folder to output resized image tiles to')
    parser.add_argument('--mode', type=str, default = 'many', metavar = 'str', help='one or many, depending on which mask type to output')
    parser.add_argument('--cutwidth', type=int, default=256, metavar='N', help='cut input into tiles of this width')
    parser.add_argument('--resizewidth', type=int, default=32, metavar='N', help='resize the tiles to this width')
    args = parser.parse_args()
    if not os.path.exists(args.data_folder):
        raise("data folder {} does not exist".format(args.data_folder))
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    IMG_FOLDER = args.data_folder + "images/"
    LABEL_FOLDER = args.data_folder + "labels/"
    MASK_FOLDER = args.data_folder + "masks/"
    if not os.path.exists(MASK_FOLDER):
        os.mkdir(MASK_FOLDER)
    keywords = ["post"]
    if args.mode == 'one':
        for file in tqdm(os.listdir(LABEL_FOLDER)):
            if file.endswith(".json"):
                filename = os.path.splitext(file)[0]
                filename_fragments = filename.split("_")
                samekeywords = list(set(filename_fragments) & set(keywords)) #um nur die post jsons auszuwählen
                if len(samekeywords) == len(keywords):
                    #print(LABEL_FOLDER)
                    #print(file)
                    load.create_mask_png(LABEL_FOLDER,file,MASK_FOLDER)
                    augment.cut_and_resize(IMG_FOLDER,file[:-19]+"_pre_disaster.png",file[:-5]+".png",MASK_FOLDER,file[:-19]+"_mask.png",args.output_folder,args.output_folder,args.cutwidth,args.resizewidth)
    if args.mode == 'many':
        for file in tqdm(os.listdir(LABEL_FOLDER)):
            if file.endswith(".json"):
                filename = os.path.splitext(file)[0]
                filename_fragments = filename.split("_")
                samekeywords = list(set(filename_fragments) & set(keywords)) #um nur die post jsons auszuwählen
                if len(samekeywords) == len(keywords):
                    #print(LABEL_FOLDER)
                    #print(file)
                    load.create_multiple_mask_pngs(LABEL_FOLDER,file,MASK_FOLDER)
                    augment.cut_and_resize_one(IMG_FOLDER,file[:-19]+"_pre_disaster.png",args.output_folder,args.cutwidth,args.resizewidth)
                    augment.cut_and_resize_one(IMG_FOLDER,file[:-5]+".png",args.output_folder,args.cutwidth,args.resizewidth)
                    for i in range(5):
                        augment.cut_and_resize_one(MASK_FOLDER,file[:-19]+"_mask%s.png" % i,args.output_folder,args.cutwidth,args.resizewidth)
    if args.mode == 'nocut':
        for file in tqdm(os.listdir(LABEL_FOLDER)):
            if file.endswith(".json"):
                filename = os.path.splitext(file)[0]
                filename_fragments = filename.split("_")
                samekeywords = list(set(filename_fragments) & set(keywords)) #um nur die post jsons auszuwählen
                if len(samekeywords) == len(keywords):
                    #print(LABEL_FOLDER)
                    #print(file)
                    load.create_multiple_mask_pngs(LABEL_FOLDER,file,MASK_FOLDER)
    if args.mode == 'manymulti':
        pool = Pool()                         # Create a multiprocessing Pool
        process_one_pars = partial(process_one, keywords = keywords, LABEL_FOLDER = LABEL_FOLDER, IMG_FOLDER = IMG_FOLDER, MASK_FOLDER = MASK_FOLDER, args = args)
        #pool.map(process_one_pars, os.listdir(LABEL_FOLDER))
        rs = pool.imap_unordered(process_one_pars, (file for file in os.listdir(LABEL_FOLDER)))
        pool.close() # No more work
        num_tasks = len(os.listdir(LABEL_FOLDER))
        while (True):
            completed = rs._index
            if (completed == num_tasks): break
            print("Waiting for", num_tasks-completed, "tasks to complete...")
            time.sleep(2)