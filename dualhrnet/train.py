import warnings

warnings.simplefilter("ignore", UserWarning)

import argparse
import multiprocessing
import os
import logging
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Variable
import torch.nn.functional as F
from yacs.config import CfgNode
from torchcontrib.optim import SWA
from tqdm import tqdm

from lovasz import lovasz_softmax
from models.dual_hrnet import get_model
from utils import AverageMeter, adjust_learning_rate
from xview2 import XView2Dataset, holdout_train, holdout2_train, holdout3_train, gupta_train
from utils import safe_mkdir, CONFIG_TREATER

disaster_list = {"ood": holdout_train, "ood2": holdout2_train, "ood3": holdout3_train, "gupta": gupta_train}

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




parser = argparse.ArgumentParser()
parser.add_argument('config_path', type=str, default="dual-hrnet",
                    help='path of model config(ex:.yaml')
parser.add_argument('--data_folder', type=str, required=True, default="",
                    help='path the data folder')
parser.add_argument("--ckpt_save_dir", type=str, default='ckpt/dual-hrnet/',
                    help='path to save checkpoints')
parser.add_argument('--mode', type=str, default = 'train',
                    help='mode')     
parser.add_argument('--swa', type =str, default = "False")           
parser.add_argument("--local_rank", type=int, default=0)

args = parser.parse_args()

ckpts_save_dir = "experiments/"+args.config_path+"/weights/"
safe_mkdir(ckpts_save_dir)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler(os.path.join(ckpts_save_dir, 'train.log')))
logger.setLevel(level=logging.DEBUG)


class ModelLossWraper(nn.Module):
    def __init__(self, model, class_weights=None, is_disaster_perd=False, is_split_loss=True):
        super(ModelLossWraper, self).__init__()
        if class_weights is None:
            class_weights = []

        self.model = model.cuda()

        self.criterion = lovasz_softmax

        self.weights = class_weights
        self.is_disaster_pred = is_disaster_perd
        self.is_split_loss = is_split_loss

    def forward(self, inputs_pre, inputs_post, targets):#, target_disaster):
        #inputs_pre = Variable(inputs_pre).cuda()
        #inputs_post = Variable(inputs_post).cuda()
        pred_dict = self.model(inputs_pre, inputs_post)
        loc = F.softmax(pred_dict['loc'], dim=1)
        loc = F.interpolate(loc, size=targets.size()[1:3], mode='bilinear')

        if self.is_split_loss:
            cls = F.softmax(pred_dict['cls'], dim=1)
            cls = F.interpolate(cls, size=targets.size()[1:3], mode='bilinear')

            targets[targets == 255] = -1

            loc_targets = targets.clone()
            loc_targets[loc_targets > 0] = 1
            loc_targets[loc_targets < 0] = 255
            #loc_targets = Variable(loc_targets).cuda()

            cls_targets = targets.clone()
            cls_targets = cls_targets - 1
            cls_targets[cls_targets < 0] = 255
            #cls_targets = Variable(cls_targets).cuda()

            # loss = self.criterion(outputs, targets, ignore_label=255)

            loc_loss = self.criterion(loc, loc_targets, ignore=255)
            cls_loss = self.criterion(cls, cls_targets, ignore=255, weights=self.weights)
            total_loss = loc_loss + cls_loss
        else:
            targets = Variable(targets).cuda()
            total_loss = self.criterion(loc, targets, ignore=255, weights=self.weights)

        #if self.is_disaster_pred:
            #target_disaster = Variable(target_disaster).cuda()
        #    disaster_loss = F.cross_entropy(pred_dict['disaster'], target_disaster)
        #    total_loss += disaster_loss * 0.05

        #print(torch.cuda.memory_summary(0))
        #print(torch.cuda.memory_summary(1))
        #print(total_loss.shape)
        #print(total_loss)
        #print(loc, loc.shape, cls, cls.shape, loc_targets, loc_targets.shape, cls_targets, cls_targets.shape)

        return total_loss


def main():
    if args.config_path:
        if args.config_path in CONFIG_TREATER:
            load_path = CONFIG_TREATER[args.config_path]
        elif args.config_path.endswith(".yaml"):
            load_path = args.config_path
        else:
            load_path = "experiments/"+CONFIG_TREATER[args.config_path]+".yaml" 
        with open(load_path, 'rb') as fp:
            config = CfgNode.load_cfg(fp)
    else:
        config = None

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    test_model = None
    max_epoch = config.TRAIN.NUM_EPOCHS
    print('data folder: ', args.data_folder)
    torch.backends.cudnn.benchmark = True

    # WORLD_SIZE Generated by torch.distributed.launch.py
    #num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    #is_distributed = num_gpus > 1
    #if is_distributed:
    #    torch.cuda.set_device(args.local_rank)
    #    torch.distributed.init_process_group(
    #        backend="nccl", init_method="env://",
    #    )

    model = get_model(config)
    model_loss = ModelLossWraper(model,
                                 config.TRAIN.CLASS_WEIGHTS,
                                 config.MODEL.IS_DISASTER_PRED,
                                 config.MODEL.IS_SPLIT_LOSS,
                                 ).cuda()

    #if args.local_rank == 0:
    #from IPython import embed; embed()

    #if is_distributed:
    #    model_loss = nn.SyncBatchNorm.convert_sync_batchnorm(model_loss)
    #    model_loss = nn.parallel.DistributedDataParallel(
    #        model_loss#, device_ids=[args.local_rank], output_device=args.local_rank
    #    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        model_loss = nn.DataParallel(model_loss)

    model_loss.to(device)
    cpucount = multiprocessing.cpu_count()

    if config.mode.startswith("single"):
        trainset_loaders = {}
        loader_len = 0
        for disaster in disaster_list[config.mode[6:]]:
            trainset = XView2Dataset(args.data_folder, rgb_bgr='rgb',
                                preprocessing={'flip': True,
                                                'scale': config.TRAIN.MULTI_SCALE,
                                                'crop': config.TRAIN.CROP_SIZE,
                                                }, mode = "singletrain", single_disaster = disaster)
            if len(trainset) > 0:
                train_sampler = None

                trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
                                                            shuffle=train_sampler is None, pin_memory=True, drop_last=True,
                                                            sampler=train_sampler, num_workers=cpucount if cpucount<16 else cpucount//3)
                
                trainset_loaders[disaster] = trainset_loader
                loader_len += len(trainset_loader)
                print("added disaster {} with {} samples".format(disaster, len(trainset)))
            else:
                print("skipping disaster ", disaster)

    else:

        trainset = XView2Dataset(args.data_folder, rgb_bgr='rgb',
                                preprocessing={'flip': True,
                                                'scale': config.TRAIN.MULTI_SCALE,
                                                'crop': config.TRAIN.CROP_SIZE,
                                                }, mode = config.mode)

        #if is_distributed:
        #    train_sampler = DistributedSampler(trainset)
        #else:
        train_sampler = None

        trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
                                                    shuffle=train_sampler is None, pin_memory=True, drop_last=True,
                                                    sampler=train_sampler, num_workers=multiprocessing.cpu_count())
        loader_len = len(trainset_loader)

    model.train()

    lr_init = config.TRAIN.LR
    optimizer = torch.optim.SGD([{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': lr_init}],
                                lr=lr_init,
                                momentum=0.9,
                                weight_decay=0.,
                                nesterov=False,
                                )

    num_iters = max_epoch * loader_len

    if config.SWA:
        swa_start = num_iters
        optimizer = SWA(optimizer, swa_start=swa_start,swa_freq = 4*loader_len, swa_lr = 0.001)#SWA(optimizer, swa_start = None, swa_freq = None, swa_lr = None)#
        #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 0.0001, 0.05, step_size_up=1, step_size_down=2*len(trainset_loader)-1, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
        lr = 0.0001
        #model.load_state_dict(torch.load("ckpt/dual-hrnet/hrnet_450", map_location='cpu')['state_dict'])
        #print("weights loaded")
        max_epoch = max_epoch+40

    start_epoch = 0
    losses = AverageMeter()
    model.train()
    cur_iters = 0 if start_epoch == 0 else None
    for epoch in range(start_epoch, max_epoch):

        if config.mode.startswith("single"):
            all_batches = []
            total_len = 0
            for disaster in sorted(list(trainset_loaders.keys())):
                all_batches += [(disaster, idx) for idx in range(len(trainset_loaders[disaster]))]
                total_len += len(trainset_loaders[disaster].dataset)
            all_batches = random.sample(all_batches, len(all_batches))
            iterators = {disaster: iter(trainset_loaders[disaster]) for disaster in trainset_loaders.keys()}
            if cur_iters is not None:
                cur_iters += len(all_batches)
            else:
                cur_iters = epoch * len(all_batches)

            for i, (disaster, idx) in enumerate(all_batches):
                lr = optimizer.param_groups[0]['lr']
                if not config.SWA or epoch < swa_start:
                    lr = adjust_learning_rate(optimizer, lr_init, num_iters, i + cur_iters)
                samples = next(iterators[disaster])
                inputs_pre = samples['pre_img'].to(device)
                inputs_post = samples['post_img'].to(device)
                target = samples['mask_img'].to(device)
                #disaster_target = samples['disaster'].to(device)

                loss = model_loss(inputs_pre, inputs_post, target)#, disaster_target)

                loss_sum = torch.sum(loss).detach().cpu()
                if np.isnan(loss_sum) or np.isinf(loss_sum):
                    print('check')
                losses.update(loss_sum, 4)  # batch size

                loss = torch.sum(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                if args.local_rank == 0 and i % 10 == 0:
                    logger.info('epoch: {0}\t'
                                'iter: {1}/{2}\t'
                                'lr: {3:.6f}\t'
                                'loss: {loss.val:.4f} ({loss.ema:.4f})\t'
                                'disaster: {dis}'.format(
                        epoch + 1, i + 1, len(all_batches), lr, loss=losses, dis=disaster))

            del iterators

        else:
            cur_iters = epoch * len(trainset_loader)

            for i, samples in enumerate(trainset_loader):
                lr = optimizer.param_groups[0]['lr']
                if not config.SWA or epoch < swa_start:
                    lr = adjust_learning_rate(optimizer, lr_init, num_iters, i + cur_iters)

                inputs_pre = samples['pre_img'].to(device)
                inputs_post = samples['post_img'].to(device)
                target = samples['mask_img'].to(device)
                #disaster_target = samples['disaster'].to(device)

                loss = model_loss(inputs_pre, inputs_post, target)#, disaster_target)

                loss_sum = torch.sum(loss).detach().cpu()
                if np.isnan(loss_sum) or np.isinf(loss_sum):
                    print('check')
                losses.update(loss_sum, 4)  # batch size

                loss = torch.sum(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                #if args.swa == "True":
                    #scheduler.step()
                    #if epoch%4 == 3 and i == len(trainset_loader)-2:
                    #    optimizer.update_swa()

                if args.local_rank == 0 and i % 10 == 0:
                    logger.info('epoch: {0}\t'
                                'iter: {1}/{2}\t'
                                'lr: {3:.6f}\t'
                                'loss: {loss.val:.4f} ({loss.ema:.4f})'.format(
                        epoch + 1, i + 1, len(trainset_loader), lr, loss=losses))

        if args.local_rank == 0:
            if (epoch + 1) % 50 == 0 and test_model is None:
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(ckpts_save_dir, 'hrnet_%s' % (epoch + 1)))
    if config.SWA:
        torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(ckpts_save_dir, 'hrnet_%s' % ("preSWA")))
        optimizer.swap_swa_sgd()
        bn_loader = torch.utils.data.DataLoader(trainset, batch_size=2,
                                                  shuffle=train_sampler is None, pin_memory=True, drop_last=True,
                                                  sampler=train_sampler, num_workers=multiprocessing.cpu_count())
        bn_update(bn_loader, model, device='cuda')
        torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(ckpts_save_dir, 'hrnet_%s' % ("SWA")))

if __name__ == '__main__':
    #multiprocessing.set_start_method('spawn', True)
    main()
