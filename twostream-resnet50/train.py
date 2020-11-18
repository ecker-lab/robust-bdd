"""Training the models

Feed model settings via a .json. Feed optional checkpoint via a .pt.

"""

import argparse
import json
import multiprocessing
import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as tr 
from torchcontrib.optim import SWA

from model.twostream_resnet50_diff import TwoStream_Resnet50_Diff

from model.caller import train, test
from utils.dataset import DisastersDatasetUnet, holdout_train, holdout2_train, holdout3_train, gupta_train, gupta_test, big_train

default_dict = {
        "model": {
            "type": "onestage",
            "name": "twostream_resnet50",
            "model_args": {"in_channels": 6, "n_classes": 5, "seperate_loss": False, "pretrained": True, "output_features": False},
        },
        "data": {
            "data_folder": ['datasets/resize128/','datasets/resize128_3/'],
            "viz_folder": ['datasets/viz/all/'],
            "holdout_folder": ['datasets/hold/all/'],
            "img_size": 128,
            "batch_size": 44,
            "disasters": "all",
            "augment_plus": True,
            "adabn": False,
            "adabn_train": False
        },
        "objective": {
            "name": "CE",
            "params": {
                "weights": [0.1197, 0.7166, 1.2869, 1.0000, 1.3640],
            }
        },
        "optimizer": {
            "name": "adam",
            "learning_rate": 0.001,
            "sheduler": {
                "patience": 2,
                "factor": 0.1
            },
            "longshedule": False
        },
        "epochs": 25,
        "seed": 42
}

model_dict = {
    "twostream_resnet50": TwoStream_Resnet50_Diff
}

disaster_dict = {
    "all": None,
    "holdout": holdout_train,
    "holdout2": holdout2_train,
    "holdout3": holdout3_train,
    "gupta": gupta_train,
    "big": big_train,
    "debug": ["lower-puna-volcano"]
}

objective_dict = {
    "CE": nn.CrossEntropyLoss
}

optimizer_dict = {
    "adam": optim.Adam,
    "SWA": optim.Adam
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train according to a given setting")
    parser.add_argument('setting', type = str, default=None, metavar='str', help='path to setting json')
    parser.add_argument('--data_folder', type = str, default = None)
    parser.add_argument('--weights', type = str, default = None ,metavar = 'str', help='path to weights pt file')
    

    args = parser.parse_args()
    setting_name = args.setting
    if setting_name is None:
        setting_path = "experiments/default_twostream_resnet50.json"
        setting_name = "default_twostream_resnet50"
        with open(setting_path, 'w') as JSON:
            json.dump(default_dict, JSON)
    else:
        setting_path = "experiments/"+setting_name+".json"
    
    weight_path = args.weights

    print("Creating folders")

    Path_list = [os.path.join('experiments/',setting_name),os.path.join('experiments/',setting_name,'logs'),os.path.join('experiments/',setting_name,'weights')]
    for Path in Path_list:
        if not os.path.exists(Path):
            print("Created folder {}".format(Path))
            os.mkdir(Path)
    
    print("Loading settings")

    with open(setting_path, 'r') as JSON:
        setting_dict = json.load(JSON)

    for k, v in default_dict.items():
        if k not in setting_dict:
            setting_dict[k] = v
        else:
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if kk not in setting_dict[k]:
                        setting_dict[k][kk] = vv

    # Dataloader regime

    print("Preparing dataset")

    data_folder = setting_dict["data"]["data_folder"] if args.data_folder is None else [os.path.join(args.data_folder, f) for f in setting_dict["data"]["data_folder"]]
    img_size = setting_dict["data"]["img_size"]
    batch_size = setting_dict["data"]["batch_size"]
    disaster_list = disaster_dict[setting_dict["data"]["disasters"]]
    augment_plus = setting_dict["data"]["augment_plus"]
    cpu_count = multiprocessing.cpu_count()
    cpu_count = cpu_count if cpu_count < 16 else cpu_count//8


    dataset = DisastersDatasetUnet(data_folder, train=True, im_size=[img_size, img_size], transform=tr.ToTensor(),normalize=False,flip = True, rotate = augment_plus, rotate10 = augment_plus, color = augment_plus * (0.2 if setting_dict["data"]["disasters"] == "big" else 1), cut = (setting_dict["data"]["disasters"] == "big" and img_size == 1024), disaster_list = disaster_list)

    if setting_dict["data"]["disasters"] == "big":
        train_size = int(0.95 * dataset.dataset_size)
    else:
        train_size = int(0.8 * dataset.dataset_size)
    val_size = dataset.dataset_size - train_size

    
    torch.manual_seed(setting_dict["seed"])
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    val_dataset.dataset.flip = False
    val_dataset.dataset.rotate = False
    val_dataset.dataset.rotate10 = False
    val_dataset.dataset.color = False

    if not setting_dict["data"]["adabn_train"]:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count, pin_memory = True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=cpu_count, pin_memory = True)
        dataloaders = {'train': train_loader, 'val': val_loader}
    else:
        train_loaders = {}
        val_loaders = {}
        for disaster in disaster_list:
            cur_subset = deepcopy(train_dataset)
            #import IPython; IPython.embed(); exit(1)
            cur_set = deepcopy(cur_subset.dataset)
            cur_list = [cur_set._DisastersDatasetUnet__mask0[i] for i in cur_subset.indices]
            filt_list = [f for f in cur_list if disaster in f]
            cur_set._DisastersDatasetUnet__mask0 = filt_list
            cur_set._DisastersDatasetUnet__mask1 = [f.replace("mask0","mask1") for f in filt_list]
            cur_set._DisastersDatasetUnet__mask2 = [f.replace("mask0","mask2") for f in filt_list]
            cur_set._DisastersDatasetUnet__mask3 = [f.replace("mask0","mask3") for f in filt_list]
            cur_set._DisastersDatasetUnet__mask4 = [f.replace("mask0","mask4") for f in filt_list]
            cur_set._DisastersDatasetUnet__pre = [f.replace("_mask0","_pre_disaster") for f in filt_list]
            cur_set._DisastersDatasetUnet__post = [f.replace("_mask0","_post_disaster") for f in filt_list]
            cur_set.dataset_size = len(filt_list)

            train_loaders[disaster] = DataLoader(cur_set, batch_size=batch_size, shuffle=True, num_workers=cpu_count, pin_memory = True)

            cur_subset = deepcopy(val_dataset)
            cur_set = deepcopy(cur_subset.dataset)
            cur_list = [cur_set._DisastersDatasetUnet__mask0[i] for i in cur_subset.indices]
            filt_list = [f for f in cur_list if disaster in f]
            cur_set._DisastersDatasetUnet__mask0 = filt_list
            cur_set._DisastersDatasetUnet__mask1 = [f.replace("mask0","mask1") for f in filt_list]
            cur_set._DisastersDatasetUnet__mask2 = [f.replace("mask0","mask2") for f in filt_list]
            cur_set._DisastersDatasetUnet__mask3 = [f.replace("mask0","mask3") for f in filt_list]
            cur_set._DisastersDatasetUnet__mask4 = [f.replace("mask0","mask4") for f in filt_list]
            cur_set._DisastersDatasetUnet__pre = [f.replace("_mask0","_pre_disaster") for f in filt_list]
            cur_set._DisastersDatasetUnet__post = [f.replace("_mask0","_post_disaster") for f in filt_list]
            cur_set.dataset_size = len(filt_list)

            val_loaders[disaster] = DataLoader(cur_set, batch_size=batch_size, shuffle=False, num_workers=cpu_count, pin_memory = True)
        
        dataloaders = {'train': train_loaders, 'val': val_loaders}
        
            

        

    # Model

    print("Initializing model")

    model_name = setting_dict["model"]["name"]
    model_args = setting_dict["model"]["model_args"]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(setting_dict["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    model = model_dict[model_name](**model_args)

    if weight_path is not None:
            model.eval()
            try:
                model.load_state_dict(torch.load(weight_path)["state_dict"])
                print("Loaded pretrained weights from file!")
                loaded = True
            except:
                loaded = False

    print("dataparallel")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)

    if weight_path is not None:
            if not loaded:
                print("loading weights")
                model.eval()
                model.load_state_dict(torch.load(weight_path)["state_dict"])
                print("Loaded pretrained weights from file!")
    
    

    # Objective
    
    print("Setting Objective function")

    objective_name = setting_dict['objective']['name']
    objective_params = setting_dict['objective']['params']

    if objective_name == "CE":
        class_weights = torch.FloatTensor(objective_params["weights"]).to(device)
        objective = objective_dict[objective_name](weight = class_weights)
    else:
        raise ValueError('objective not supported yet')

    # Optimizer

    print("Initializing Optimizer")

    optimizer_name = setting_dict['optimizer']['name']
    lr = setting_dict['optimizer']['learning_rate']

    if optimizer_name == "SWA":
        base_opt = optimizer_dict[optimizer_name](model.parameters(), lr=lr)
        if not setting_dict["data"]["adabn_train"]:
            steps_per_epoch = len(train_loader)
        else:
            steps_per_epoch = sum([len(dataloader) for dataloader in train_loaders])
        swa_kick = setting_dict['optimizer']['swa_kick']
        #optimizer = SWA(base_opt, swa_start=100, swa_freq=50, swa_lr=lr/2)
        optimizer = SWA(base_opt, swa_start=swa_kick*steps_per_epoch, swa_freq=steps_per_epoch, swa_lr=lr/2)
    else:
        if model_name == "loc_wrapper":
            optimizer = optimizer_dict[optimizer_name](model.dmg_model.parameters(), lr=lr)
        else:
            optimizer = optimizer_dict[optimizer_name](model.parameters(), lr=lr)

    # Call

    print("Starting model training....")

    n_epochs = setting_dict['epochs']
    lr_patience = setting_dict['optimizer']['sheduler']['patience']
    lr_factor = setting_dict['optimizer']['sheduler']['factor']

    if weight_path is None:
        best_epoch = train(model,dataloaders,objective,optimizer,n_epochs,Path_list[1],Path_list[2], lr_patience=lr_patience,lr_factor=lr_factor, dice = False,seperate_loss=False, adabn = setting_dict["data"]["adabn_train"], own_sheduler = (not setting_dict["optimizer"]["longshedule"]))
    else:
        optimizer.load_state_dict(torch.load(weight_path)["optimizer"])
        best_epoch = train(model,dataloaders,objective,optimizer,n_epochs-torch.load(weight_path)["epoch"],Path_list[1],Path_list[2],start_epoch = torch.load(weight_path)["epoch"]+1, loss_dict=torch.load(weight_path)["loss_dict"], lr_patience=lr_patience,lr_factor=lr_factor, dice = False,seperate_loss=False, adabn = setting_dict["data"]["adabn_train"], own_sheduler = (not setting_dict["optimizer"]["longshedule"]))

    print("model training finished! yey!")

    if optimizer_name == "SWA":
        print ("Updating batch norm pars for SWA")
        train_dataset.dataset.SWA = True
        SWA_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count)
        optimizer.swap_swa_sgd()
        optimizer.bn_update(SWA_loader, model, device='cuda')
        state = {
                'epoch': n_epochs,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss_dict': {}
                }
        torch.save(state, os.path.join(Path_list[2],'weights_SWA.pt'))
