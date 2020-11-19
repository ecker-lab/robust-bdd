import os
import argparse
import multiprocessing
import warnings
import copy
import urllib.request


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from skimage.io import imread, imsave
from yacs.config import CfgNode

from tqdm import tqdm

from models.dual_hrnet import get_model
from xview2 import XView2Dataset, holdout_test, holdout2_test, holdout3_test, gupta_test, holdout_train, holdout2_train, holdout3_train, gupta_train
from utils import safe_mkdir, download_weights, CONFIG_TREATER

from scoring.xview2_metrics import XviewMetrics


disaster_list = {"oodtest": holdout_test, "ood2test": holdout2_test, "ood3test": holdout3_test, "guptatest": gupta_test,
                    "oodhold": holdout_train, "ood2hold": holdout2_train, "ood3hold": holdout3_train, "guptahold": gupta_train}

def bn_update(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.

    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.

        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.

        device (torch.device, optional): If set, data will be trasferred to
            :attr:`device` before being passed into :attr:`model`.
    """
    if not _check_bn(model):
        return
    was_training = model.training
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    momenta = {}
    model.apply(_reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input in tqdm(loader):
        #if isinstance(input, (list, tuple)):
        #    input = input[0]
        inputs_pre = input['pre_img']
        inputs_post = input['post_img']
        b = inputs_pre.size(0)

        momentum = b / float(n + b)
        for module in momenta.keys():
            module.momentum = momentum

        #if device is not None:
        #    input = input.to(device)

        model(inputs_pre.to(device), inputs_post.to(device))
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))
    model.train(was_training)


# BatchNorm utils
def _check_bn_apply(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def _check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn_apply(module, flag))
    return flag[0]


def _reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]




multiprocessing.set_start_method('spawn', True)
warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('config_path', type=str, default="",
                    help='the location of the data folder')
parser.add_argument('--data_folder', type=str, required=True, default="/mnt/Dataset/xView2/v2",
                    help='the location of the data folder')
parser.add_argument('--weights', type=str, required=True, default="",
                    help='Path to checkpoint')
#parser.add_argument('--result_dir', type=str, required=True, default="",
#                    help='Path to save result submit and compare iamges')
parser.add_argument('--dataset_mode', default = "test",
                    help='')
parser.add_argument('--is_use_gpu', action='store_true', dest='is_use_gpu',
                    help='')

args = parser.parse_args()


class ModelWraper(nn.Module):
    def __init__(self, model, is_use_gpu=False, is_split_loss=True):
        super(ModelWraper, self).__init__()
        self.is_use_gpu = is_use_gpu
        self.is_split_loss = is_split_loss
        if self.is_use_gpu:
            self.model = model.cuda()
        else:
            self.model = model

    def forward(self, inputs_pre, inputs_post):
        inputs_pre = Variable(inputs_pre)
        inputs_post = Variable(inputs_post)

        if self.is_use_gpu:
            inputs_pre = inputs_pre.cuda()
            inputs_post = inputs_post.cuda()

        pred_dict = self.model(inputs_pre, inputs_post)
        loc = F.interpolate(pred_dict['loc'], size=inputs_pre.size()[2:4], mode='bilinear')

        if self.is_split_loss:
            cls = F.interpolate(pred_dict['cls'], size=inputs_post.size()[2:4], mode='bilinear')
        else:
            cls = None

        return loc, cls


def argmax(loc, cls):
    loc = torch.argmax(loc, dim=1, keepdim=False)
    cls = torch.argmax(cls, dim=1, keepdim=False)

    cls = cls + 1
    cls[loc == 0] = 0

    return loc, cls


def main():
    if args.config_path:
        if args.config_path in CONFIG_TREATER:
            with open("experiments/"+CONFIG_TREATER[args.config_path]+".yaml", 'rb') as fp:
                config = CfgNode.load_cfg(fp)
        else:
            with open("experiments/"+args.config_path+".yaml", 'rb') as fp:
                config = CfgNode.load_cfg(fp)
    else:
        config = None

    ckpt_path = args.weights
    if ckpt_path == "paper":
        ckpt_path = download_weights(args.config_path)
    
    result_submit_dir = "experiments/"+args.config_path+"/output/"#args.result_dir #os.path.join(args.result_dir, 'submit/')
    #result_compare_dir = os.path.join(args.result_dir, 'compare/')

    
    #imgs_dir = os.path.join(args.data_path, 'test/images/')if dataset_mode == 'test' \
    #    else os.path.join(args.data_path, 'tier3/images/') 

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    print('data folder: ', args.data_folder)

    

    model = get_model(config)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['state_dict'])
    model.eval()
    model_wrapper = ModelWraper(model, not args.is_use_gpu, config.MODEL.IS_SPLIT_LOSS)
    # model_wrapper = nn.DataParallel(model_wrapper)
    model_wrapper.eval()

    for dataset_mode in [config.mode+"test", config.mode+"hold"]:
        result_submit_dir = "experiments/"+args.config_path+"/output/"+dataset_mode
        os.makedirs(result_submit_dir, exist_ok = True)
        safe_mkdir(result_submit_dir)
        safe_mkdir(os.path.join(result_submit_dir,"predictions"))
        safe_mkdir(os.path.join(result_submit_dir,"targets"))
        #safe_mkdir(result_compare_dir)
        if dataset_mode.startswith("single"):
            with torch.no_grad():
                testset_loaders = {}
                for disaster in disaster_list[dataset_mode[6:]]:
                    testmode = "singletest" if "test" in dataset_mode else "singlehold"
                    testset = XView2Dataset(args.data_folder, rgb_bgr='rgb', preprocessing={'flip': False, 'scale': None, 'crop': None}, mode = testmode, single_disaster = disaster)
                    if len(testset) > 0:
                        print("added disaster {} with {} samples".format(disaster, len(testset)))
                        testset_loader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, pin_memory=False, num_workers=4)
                    
                        testset_loaders[disaster] = testset_loader
                    else:
                        print("skipped disaster ", disaster)

                for disaster in sorted(list(testset_loaders.keys())):
                    loader = testset_loaders[disaster]
                    if len(loader) == 0:
                        continue
                    print(disaster)
                    bn_update(loader, model, device='gpu')
                    #model_wrapper = ModelWraper(model, args.is_use_gpu, config.MODEL.IS_SPLIT_LOSS)
                    #model_wrapper.eval()
                    for i, samples in enumerate(tqdm(loader)):
                        if dataset_mode == 'train' and i < 5520:
                            continue
                        inputs_pre = samples['pre_img']
                        inputs_post = samples['post_img']
                        image_ids = samples['image_id']
                        if dataset_mode[6:] in ["oodtest","oodhold","ood2test","ood2hold","ood3test","ood3hold","guptatest","guptahold"]:
                            masks = samples['mask_img']

                        loc, cls = model_wrapper(inputs_pre, inputs_post)

                        if config.MODEL.IS_SPLIT_LOSS:
                            loc, cls = argmax(loc, cls)
                            loc = loc.detach().cpu().numpy().astype(np.uint8)
                            cls = cls.detach().cpu().numpy().astype(np.uint8)
                        else:
                            loc = torch.argmax(loc, dim=1, keepdim=False)
                            loc = loc.detach().cpu().numpy().astype(np.uint8)
                            cls = copy.deepcopy(loc)

                        for i, (image_id, l, c) in enumerate(zip(image_ids, loc, cls)):
                            localization_filename = 'test_localization_%s_prediction.png' % image_id
                            damage_filename = 'test_damage_%s_prediction.png' % image_id

                            imsave(os.path.join(result_submit_dir, "predictions", localization_filename), l)
                            imsave(os.path.join(result_submit_dir, "predictions", damage_filename), c)

                            if dataset_mode[6:] in ["oodtest","oodhold","ood2test","ood2hold","ood3test","ood3hold","guptatest","guptahold"]:
                                localization_filename = 'test_localization_%s_target.png' % image_id
                                damage_filename = 'test_damage_%s_target.png' % image_id
                                
                                mask = masks[i]
                                mask[mask == 255] = 0
                                mask = mask.cpu().numpy().astype(np.uint8)

                                imsave(os.path.join(result_submit_dir, "targets", localization_filename), (1*(mask > 0)))
                                imsave(os.path.join(result_submit_dir, "targets", damage_filename), mask)
                    #model_wrapper.model.cpu()

        else:
            testset = XView2Dataset(args.data_folder, rgb_bgr='rgb', preprocessing={'flip': False, 'scale': None, 'crop': None},
                                mode=dataset_mode)
            testset_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, pin_memory=False, num_workers=1)

            for i, samples in enumerate(tqdm(testset_loader)):
                if i > 10:
                    break
                if dataset_mode == 'train' and i < 5520:
                    continue
                inputs_pre = samples['pre_img']
                inputs_post = samples['post_img']
                image_ids = samples['image_id']
                if dataset_mode in ["oodtest","oodhold","ood2test","ood2hold","ood3test","ood3hold","guptatest","guptahold"]:
                    masks = samples['mask_img']

                loc, cls = model_wrapper(inputs_pre, inputs_post)

                if config.MODEL.IS_SPLIT_LOSS:
                    loc, cls = argmax(loc, cls)
                    loc = loc.detach().cpu().numpy().astype(np.uint8)
                    cls = cls.detach().cpu().numpy().astype(np.uint8)
                else:
                    loc = torch.argmax(loc, dim=1, keepdim=False)
                    loc = loc.detach().cpu().numpy().astype(np.uint8)
                    cls = copy.deepcopy(loc)

                for i, (image_id, l, c) in enumerate(zip(image_ids, loc, cls)):
                    localization_filename = 'test_localization_%s_prediction.png' % image_id
                    damage_filename = 'test_damage_%s_prediction.png' % image_id

                    imsave(os.path.join(result_submit_dir, "predictions", localization_filename), l)
                    imsave(os.path.join(result_submit_dir, "predictions", damage_filename), c)

                    if dataset_mode in ["oodtest","oodhold","ood2test","ood2hold","ood3test","ood3hold","guptatest","guptahold"]:
                        localization_filename = 'test_localization_%s_target.png' % image_id
                        damage_filename = 'test_damage_%s_target.png' % image_id
                        
                        mask = masks[i]
                        mask[mask == 255] = 0
                        mask = mask.cpu().numpy().astype(np.uint8)

                        imsave(os.path.join(result_submit_dir, "targets", localization_filename), (1*(mask > 0)))
                        imsave(os.path.join(result_submit_dir, "targets", damage_filename), mask)


        base = result_submit_dir
        if True:
            for i, p in enumerate(os.listdir(os.path.join(base,"predictions"))):
                if "damage" in p:
                    os.rename(os.path.join(base,"targets",p.replace("prediction", "target")), os.path.join(base,"targets", "_".join(p.split("_")[:2]+[str(i).zfill(6)]+["target.png"])))
                    os.rename(os.path.join(base,"predictions",p), os.path.join(base,"predictions","_".join(p.split("_")[:2]+[str(i).zfill(6)]+[p.split("_")[-1]])))
                    p = p.replace("damage","localization")
                    os.rename(os.path.join(base,"targets",p.replace("prediction", "target")), os.path.join(base,"targets","_".join(p.split("_")[:2]+[str(i).zfill(6)]+["target.png"])))
                    os.rename(os.path.join(base,"predictions",p), os.path.join(base,"predictions","_".join(p.split("_")[:2]+[str(i).zfill(6)]+[p.split("_")[-1]])))


        MetricsInstance = XviewMetrics(os.path.join(base,"predictions"), os.path.join(base,"targets"))

        MetricsInstance.compute_score(os.path.join(base,"predictions"), os.path.join(base,"targets"), os.path.join(base,"Results_{}.json".format(dataset_mode)))


if __name__ == '__main__':
    main()
