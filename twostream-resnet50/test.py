"""Evaluating the models

Feed first model settings as .json dict. Optionally also feed checkpoints and a second model (for two-stage approaches).

"""

import argparse
import json
import multiprocessing
import os
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as tr 

from model.twostream_resnet50_diff import TwoStream_Resnet50_Diff

from model.caller import train, test, test_twostage
from utils.dataset import DisastersDatasetUnet, holdout_train, holdout_test, holdout2_train, holdout2_test, holdout3_train, holdout3_test, gupta_train, gupta_test, big_train, big_test
from utils.xview2_metrics import XviewMetrics
from utils.download import download_weights

from train import default_dict, model_dict


disaster_dict = {
    "all": (None, None),
    "holdout": (holdout_train, holdout_test),
    "holdout2": (holdout2_train, holdout2_test),
    "holdout3": (holdout3_train, holdout3_test),
    "gupta": (gupta_train, gupta_test),
    "big": (big_train, big_test),
    "hold": (big_train, big_test)
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train according to a given setting")
    parser.add_argument('setting', type = str, metavar='setting.json', help='path to setting json')
    parser.add_argument('--weights', type = str, default = None ,metavar = 'str', help='path to weights pt file')
    parser.add_argument('--mode', type = str, default = "both", metavar='test hold both', help='mode')
    parser.add_argument('--second_model', type = str, default = None, metavar='setting.json', help='path to setting json of second model')
    parser.add_argument('--second_model_weights', type = str, default = None ,metavar = 'str', help='path to weights pt file')
    parser.add_argument('--data_folder', type = str, default = None)

    args = parser.parse_args()

    setting_name = args.setting
    setting_name_components = setting_name.split("_")
    if setting_name_components[2].startswith("multi"):
        setting_name_components[2] = "multi"
    else:
        setting_name_components[2] = "plain"
    setting_path = "experiments/"+"_".join(setting_name_components)+".json"

    second_model_path = args.second_model
    if args.weights == "paper":
        weight_path = download_weights(setting_name)
    else:
        weight_path = args.weights
    second_weight_path = args.second_model_weights

    print("Creating folders")

    Path_list = [os.path.join('experiments',setting_name),os.path.join('experiments',setting_name,'output'),os.path.join('experiments',setting_name,'output','val'),os.path.join('experiments',setting_name,'output','viz'),os.path.join('experiments',setting_name,'output','test'),os.path.join('experiments',setting_name,'output','val','images'),os.path.join('experiments',setting_name,'output','val','predictions'),os.path.join('experiments',setting_name,'output','val','targets'),os.path.join('experiments',setting_name,'output','viz','images'),os.path.join('experiments',setting_name,'output','viz','predictions'),os.path.join('experiments',setting_name,'output','viz','targets'),os.path.join('experiments',setting_name,'output','test','images'),os.path.join('experiments',setting_name,'output','test','predictions'),os.path.join('experiments',setting_name,'output','test','targets'),os.path.join('experiments',setting_name,'output','hold'),os.path.join('experiments',setting_name,'output','hold','images'),os.path.join('experiments',setting_name,'output','hold','targets'),os.path.join('experiments',setting_name,'output','hold','predictions')]
    #Path_list = [os.path.join('/local/jobs/5314827/', p) for p in Path_list]
    for Path in Path_list + [os.path.join('experiments',setting_name)]:
        if not os.path.exists(Path):
            print("Created folder {}".format(Path))
            os.mkdir(Path)
    
    print("Loading settings")

    with open(setting_path, 'r') as JSON:
        setting_dict = json.load(JSON)

    if second_model_path is not None:
        with open(second_model_path, 'r') as JSON:
            second_dict = json.load(JSON)

    for k, v in default_dict.items():
        if k not in setting_dict:
            setting_dict[k] = v
        else:
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if kk not in setting_dict[k]:
                        setting_dict[k][kk] = vv

    model_type = setting_dict["model"]["type"]

    if model_type in ["twostageloc","twostagedmg"]:
        model_number = 2
    else:
        model_number = 1

    # Dataloader regime

    print("Preparing dataset")

    data_folder = setting_dict["data"]["data_folder"] if args.data_folder is None else [os.path.join(args.data_folder, f) for f in setting_dict["data"]["data_folder"]]
    holdout_folder = setting_dict["data"]["holdout_folder"]if args.data_folder is None else [os.path.join(args.data_folder, f) for f in setting_dict["data"]["holdout_folder"]]
    viz_folder = setting_dict["data"]["viz_folder"] if args.data_folder is None else [os.path.join(args.data_folder, f) for f in setting_dict["data"]["viz_folder"]]
    img_size = setting_dict["data"]["img_size"]
    batch_size = setting_dict["data"]["batch_size"]
    disaster_list_train, disaster_list_test = disaster_dict[setting_dict["data"]["disasters"]]
    cpu_count = multiprocessing.cpu_count()

    train_dataset = DisastersDatasetUnet(data_folder, train=True, im_size=[img_size, img_size], transform=tr.ToTensor(),normalize=False,flip = True, disaster_list = disaster_list_train)

    if setting_dict["data"]["disasters"] == "big":
        train_size = int(0.95 * train_dataset.dataset_size)
    else:
        train_size = int(0.8 * train_dataset.dataset_size)
    val_size = train_dataset.dataset_size - train_size

    torch.manual_seed(setting_dict["seed"])
    _, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    val_dataset.dataset.flip = False

    if not setting_dict["data"]["adabn"]:

        test_dataset = DisastersDatasetUnet(data_folder, train=True, im_size=[img_size, img_size], transform=tr.ToTensor(),normalize=False, flip = False, disaster_list = disaster_list_test)
        print(disaster_list_test)
        print(set([d.split("_")[0] for d in test_dataset.files]))
        hold_dataset = DisastersDatasetUnet(holdout_folder, train=True, im_size=[1024,1024], transform=tr.ToTensor(),normalize=False, flip = False, disaster_list = disaster_list_train)
        print(disaster_list_train)
        print(set([d.split("_")[0] for d in hold_dataset.files]))
        viz_dataset = DisastersDatasetUnet(viz_folder, train=True, im_size=[1024,1024], transform=tr.ToTensor(),normalize=False, flip = False)

        batchsizefactor = 4 if model_number == 1 else 2

        val_loader = DataLoader(val_dataset, batch_size=batchsizefactor*batch_size, shuffle=False, num_workers=cpu_count)
        test_loader = DataLoader(test_dataset, batch_size=batchsizefactor*batch_size, shuffle=False, num_workers=cpu_count)
        hold_loader = DataLoader(hold_dataset, batch_size=max(1,int(batch_size*(batchsizefactor/16))), shuffle=False, num_workers=cpu_count)
        viz_loader = DataLoader(viz_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    else:
        batchsizefactor = 4 if model_number == 1 else 2
        test_loaders = []
        for disaster in disaster_list_test:
            test_dataset = DisastersDatasetUnet(data_folder, train=True, im_size=[img_size, img_size], transform=tr.ToTensor(),normalize=False, flip = False, disaster_list = [disaster])
            print(disaster)
            print(set([d.split("_")[0] for d in test_dataset.files]))
            test_loaders.append(DataLoader(test_dataset, batch_size=batchsizefactor*batch_size, shuffle=False, num_workers=cpu_count))
        hold_loaders = []
        for disaster in disaster_list_train:
            hold_dataset = DisastersDatasetUnet(holdout_folder, train=True, im_size=[1024, 1024], transform=tr.ToTensor(),normalize=False, flip = False, disaster_list = [disaster])
            print(disaster)
            print(set([d.split("_")[0] for d in hold_dataset.files]))
            hold_loaders.append(DataLoader(hold_dataset, batch_size=max(1,int(batch_size*(batchsizefactor/16))), shuffle=False, num_workers=cpu_count))
        viz_dataset = DisastersDatasetUnet(viz_folder, train=True, im_size=[1024,1024], transform=tr.ToTensor(),normalize=False, flip = False)
        val_loader = DataLoader(val_dataset, batch_size=batchsizefactor*batch_size, shuffle=False, num_workers=cpu_count)
        viz_loader = DataLoader(viz_dataset, batch_size=1, shuffle=False, num_workers=1)

    # Model

    print("Initializing model(s)")

    if model_number == 1:

        model_name = setting_dict["model"]["name"]
        model_args = setting_dict["model"]["model_args"]
        
        print("device")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("seed")
        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("load model")
        model = model_dict[model_name](**model_args)
        print("load weights")

        if weight_path is not None:
            model.eval()
            try:
                model.load_state_dict(torch.load(weight_path)["state_dict"])
                loaded = True
            except:
                loaded = False
            
        print("dataparallel")
        if torch.cuda.device_count() > 1 or ((weight_path is not None) and (not loaded)):
            model = nn.DataParallel(model)

        if weight_path is not None:
            if not loaded:
                model.eval()
                model.load_state_dict(torch.load(weight_path)["state_dict"])
            
        print("to gpu")
        model.to(device)
        
        
        print("set tester")
        model_tester = partial(test, model = model)


    elif model_number == 2:
        fusion_style = 'none'
        models = {}
        for stage in ["twostageloc","twostagedmg"]:
            if model_type == stage:
                current_dict = setting_dict
                current_weight = weight_path
            elif second_model_path is not None:
                current_dict = second_dict
                current_weight = second_weight_path
            else:
                if stage == "twostageloc":
                    current_dict = {"model": {"name": "unet", "model_args": {"in_channels": 6, "n_classes": 1, "depth": 2, "wf": 3, "padding": True, "batch_norm": True, "dropout": False, "up_mode": "upsample", "seperate_loss": False}}}
                    fusion_style = 'dmg'
                elif stage == "twostagedmg":
                    current_dict = {"model": {"name": "unet", "model_args": {"in_channels": 6, "n_classes": 5, "depth": 2, "wf": 3, "padding": True, "batch_norm": True, "dropout": False, "up_mode": "upsample", "seperate_loss": False}}}
                current_weight = None
            current_name = current_dict["model"]["name"]
            current_args = current_dict["model"]["model_args"]

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            torch.manual_seed(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            current_model = model_dict[current_name](**current_args)
            current_model.to(device)

            if current_weight is not None:
                current_model.eval()
                current_model.load_state_dict(torch.load(current_weight)["state_dict"])

            models[stage] = current_model
        
        model_tester = partial(test_twostage, loc_model = models["twostageloc"], dmg_model = models["twostagedmg"], fusion_style = fusion_style)

    else:
        raise("This is the weirdest error ever because you somehow managed to give a different number of models to the script which is by definition of the model number impossible.")


    # Viz prediction


    print("Prediction on small viz dataset (for debugging purposes) started")

    #model_tester(dataloader = viz_loader, output_path_directory = Path_list[3])

    # Val prediction

    print("Prediction on validation dataset started")

    #model_tester(dataloader = val_loader, output_path_directory = Path_list[2])
    
    MetricsInstance = XviewMetrics(Path_list[6],Path_list[7])

    #MetricsInstance.compute_score(Path_list[6],Path_list[7], os.path.join(Path_list[0],"Results_val.json"))

    # Test prediction

    print("Prediction on test dataset started")

    if args.mode in ["test","both"]:
        if not setting_dict["data"]["adabn"]:
            model_tester(dataloader = test_loader, output_path_directory = Path_list[4], adapt = False)
        else:
            for idx, dataloader in enumerate(test_loaders):
                #base_path = os.path.join(Path_list[4],disaster_list_test[idx])
                #os.makedirs(base_path, exist_ok = True)
                #os.makedirs(os.path.join(base_path,"images"))
                #os.makedirs(os.path.join(base_path,"targets"))
                #os.makedirs(os.path.join(base_path,"predictions"))
                #model_tester(dataloader = dataloader, output_path_directory = base_path, adapt = False, start_idx = idx*200000)
                #MetricsInstance.compute_score(os.path.join(base_path,"predictions"),os.path.join(base_path,"targets"), os.path.join('experiments',setting_name,"Results_test_{}.json".format(disaster_list_test[idx])))
                model_tester(dataloader = dataloader, output_path_directory = Path_list[4], adapt = True, start_idx = idx*200000)

        MetricsInstance.compute_score(Path_list[12],Path_list[13], os.path.join('experiments',setting_name,"Results_OOD.json"))

    # Hold prediction

    print("Prediction on holdout dataset started")
    if args.mode in ["hold","both"]:
        if not setting_dict["data"]["adabn"]:
            model_tester(dataloader = hold_loader, output_path_directory = Path_list[14], adapt = False)
        else:
            for idx, dataloader in enumerate(hold_loaders):
                #base_path = os.path.join(Path_list[14],disaster_list_train[idx])
                #os.makedirs(base_path, exist_ok = True)
                #os.makedirs(os.path.join(base_path,"images"))
                #os.makedirs(os.path.join(base_path,"targets"))
                #os.makedirs(os.path.join(base_path,"predictions"))
                if len(dataloader) > 0:
                    #model_tester(dataloader = dataloader, output_path_directory = base_path, adapt = False, start_idx = idx*200000)
                    #MetricsInstance.compute_score(os.path.join(base_path,"predictions"),os.path.join(base_path,"targets"), os.path.join('experiments',setting_name,"Results_hold_{}.json".format(disaster_list_train[idx])))
                    model_tester(dataloader = dataloader, output_path_directory = Path_list[14], adapt = True, start_idx = idx*100000)
        
        MetricsInstance.compute_score(Path_list[17],Path_list[16], os.path.join('experiments',setting_name,"Results_IID.json"))