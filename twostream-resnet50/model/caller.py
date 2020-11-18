"""Train and Test functions

These functions implement our training loop boilerplate code.
"""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as tr

from tqdm import tqdm
import scipy.io as sio
from scipy import interpolate
import numpy as np
import datetime
import json
from cv2 import fillPoly, imwrite
from PIL import Image
import os
import time
import random


from utils.load import colorize_mask_

from utils.output_fusion import max_freq_per_component_fusion

from utils.xview2_metrics import compute_tp_fn_fp, F1Recorder


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]

def adjust_learning_rate(optimizer, base_lr, max_iters, cur_iters, power=0.9):
    lr = base_lr * ((1 - float(cur_iters) / max_iters) ** power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def train(model, dataloaders, criterion, optimizer, num_epochs, output_path_loss, output_path_weights, start_epoch=1,loss_dict={},lr_patience = 5,lr_factor = 0.1,dice=False,seperate_loss=False, adabn = False, own_sheduler = True):
  """Caller function to train model

  Call this function to train a model on the xBD dataset. Creates checkpoints of weights, optimizer state, epoch number and loss dictionary at each epoch. Uses a learning rate sheduler when validation loss does not improve. Run validation on validation dataset after each epoch. Outputs loss as a json.

  Example:
    Training a TwoStream ResNet50:
    >>> model = TwoStream_Resnet50_Diff()
    >>> optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    >>> criterion = nn.CrossEntropyLoss()
    >>> dataloaders = {'train': train_loader, 'val': val_loader}
    >>> train(model, dataloaders, criterion, optimizer, 10, '/logs/', '/weights/')
  
  Args:
      model (torch.nn.Module): The PyTorch model to use
      dataloaders (dict): Is a dict with two keys 'train' and 'val' which correspond to the train resp. validation PyTorch dataloaders
      criterion (torch.nn.Module): PyTorch criterion to calculate loss
      optimizer (torch.optim.Optimizer): PyTorch optimizer
      num_epochs (int): number of epochs to train for
      output_path_loss (str): Path to output file with loss-list to
      output_path_weights (str): Path to output weights to
      start_epoch (int, optional): Epoch from which to start training, used for checkpointing. Defaults to 1.
      loss_dict (dict, optional): Loss_dict of previous epochs, needs to be aligned with start_epoch. Defaults to {}.
      lr_patience (int, optional): Patience used for learning rate sheduler, if there has been no improvement for lr_patience epochs in the validation loss, then learning rate is adjusted by factor lr_factor and weights are restored from best weights so far. Defaults to 5.
      lr_factor (float, optional): factor to adjust learning rate by if sheduler is triggered. Defaults to 0.1.
      dice (bool, optional): If using dice-loss or another loss that requires masks not to be argmaxed set to True. Defaults to False.
      seperate_loss (bool, optional): If using a model with two seperate heads for localization and damage set to True. Does not work with dice=True. Defaults to False.
  """  

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  starttime = time.time()
  starttimestamp = str(datetime.datetime.now().timestamp())
  if own_sheduler:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience = lr_patience,min_lr = 0.000001)
  else:
    base_lr = optimizer.param_groups[0]['lr']
    if not adabn:
      iters_per_epoch = len(dataloaders["train"])
      total_iters = iters_per_epoch * num_epochs
    else:
      iters_per_epoch = sum([len(dataloaders["train"][dis]) for dis in sorted(list(dataloaders["train"].keys()))])
      total_iters = iters_per_epoch * num_epochs
  best_epoch = 1
  best_val_loss = 100
  for epoch in range(start_epoch,start_epoch+num_epochs):
    if not own_sheduler:
      cur_iters = epoch * iters_per_epoch
    for phase in ['train','val']:
      if phase == 'train':
        
        model.train()
        running_loss = 0.0

        if adabn:
          all_batches = []
          total_len = 0
          for disaster in sorted(list(dataloaders[phase].keys())):
            all_batches += [(disaster, idx) for idx in range(len(dataloaders[phase][disaster]))]
            total_len += len(dataloaders[phase][disaster].dataset)
          all_batches = random.sample(all_batches, len(all_batches))

          iterators = {disaster: iter(dataloaders[phase][disaster]) for disaster in dataloaders[phase].keys()}

          for batch_idx, (disaster, idx) in enumerate(all_batches):
            if not own_sheduler:
              lr = adjust_learning_rate(optimizer, base_lr, total_iters, batch_idx + cur_iters)
            images_pre, images_post, masks, _, _ = next(iterators[disaster])
            if not dice:
              masks = torch.argmax(masks,dim=1)
            optimizer.zero_grad()

            masks = masks.to(device).long()
            images = torch.cat((images_pre, images_post),1)
            images = images.to(device) 
            if seperate_loss:
              loc, dmg = model(images)
              loss = criterion(loc, dmg, masks)
            else:
              output = model(images)
              loss = criterion(output, masks)

            loss.backward()
            optimizer.step()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}\tDisaster: {}'.format(epoch, batch_idx * len(images), total_len,100. * batch_idx / len(all_batches), loss.item(),disaster).encode('ascii','ignore'))
            #print(torch.cuda.memory_summary(0))
            #print(torch.cuda.memory_summary(1))

            running_loss += loss.item() * len(images)
          epoch_loss = running_loss / total_len
          del iterators
        
        else:
          for batch_idx, (images_pre, images_post, masks, _ , _) in enumerate(dataloaders[phase]):
            if not own_sheduler:
              lr = adjust_learning_rate(optimizer, base_lr, total_iters, batch_idx + cur_iters)
            if not dice:
              masks = torch.argmax(masks,dim=1)
            optimizer.zero_grad()

            masks = masks.to(device).long()
            images = torch.cat((images_pre, images_post),1)
            images = images.to(device) 
            if seperate_loss:
              loc, dmg = model(images)
              loss = criterion(loc, dmg, masks)
            else:
              output = model(images)
              loss = criterion(output, masks)

            loss.backward()
            optimizer.step()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(epoch, batch_idx * len(images), len(dataloaders[phase].dataset),100. * batch_idx / len(dataloaders[phase]), loss.item()).encode('ascii','ignore'))

            #print(torch.cuda.memory_summary(0))
            #print(torch.cuda.memory_summary(1))
            #print(lr)

            running_loss += loss.item() * len(images)
          epoch_loss = running_loss / len(dataloaders[phase].dataset)
          
      
      elif phase == 'val':
        lrow = np.zeros(3)
        drow = [np.zeros(3) for k in range(4)]
        with torch.no_grad():
          model.eval()
          running_loss = 0.0

          if adabn:
            total_len = 0
            for disaster in sorted(list(dataloaders[phase].keys())):
              print("Adapting {} for validation".format(disaster))
              total_len += len(dataloaders[phase][disaster].dataset)
              n = 0
              model.train()
              momenta = {}
              model.apply(lambda module: _get_momenta(module, momenta))
              for batch_idx, (images_pre, images_post, masks, _ , _) in tqdm(enumerate(dataloaders[phase][disaster]), total = len(dataloaders[phase][disaster])):
                images = torch.cat((images_pre, images_post),1)
                images = images.to(0) 
                b = images.shape[0]
                #new_imgs = []
                #for i in range(images.shape[0]):
                #  curr = images[i,...]
                #  if (1.0*(curr.sum(0) == 0)).mean() < 0.03:
                #    new_imgs.append(curr)
                #b = len(new_imgs)
                #if b > 0:
                #  images = torch.stack(new_imgs,0)
                #  images = images.to(0)
                #  limit = 128 if images.shape[-1] == 1024 else 512
                #  if len(dataloader)*images_pre.shape[0] < limit:
                #    momentum = b / float(n + b + (limit-len(dataloader)*images_pre.shape[0]))
                #  else:
                momentum = b / float(n + b)
                for module in momenta.keys():
                  module.momentum = momentum
                _ = model(images)
                n += b 
              model.eval()
              model.apply(lambda module: _set_momenta(module, momenta))
                #if len(dataloader) > 0:
                #  print(len(dataloader)*images_pre.shape[0], n)
              
              print("evaluating")
              for batch_idx, (images_pre, images_post, masks, _ , _) in tqdm(enumerate(dataloaders[phase][disaster]), total = len(dataloaders[phase][disaster])):
                if not dice:
                  masks = torch.argmax(masks,dim=1)
                masks = masks.to(device).long()
                images = torch.cat((images_pre, images_post),1)
                images = images.to(device) 
                if seperate_loss:
                  loc, dmg = model(images)
                  loss = criterion(loc, dmg, masks)
                else:
                  output = model(images)
                  loss = criterion(output, masks)

                dmg_target = masks if not dice else torch.argmax(masks,1)
                loc_target = torch.where(dmg_target > 0, torch.ones_like(dmg_target), torch.zeros_like(dmg_target))
                if output.shape[1] != 1 and len(output.shape) == 4:
                  dmg_pred = torch.argmax(output,1)
                  loc_mode = False
                  loc_pred = torch.where(dmg_pred > 0, torch.ones_like(dmg_pred), torch.zeros_like(dmg_pred))
                else:
                  loc_mode = True
                  loc_pred = (1*(output > 0.5))
                  if len(loc_pred.shape) == 4:
                    loc_pred = loc_pred[:,0,:,:]
                  
            
                lrow += compute_tp_fn_fp(loc_pred.data.byte().cpu().numpy(), loc_target.data.byte().cpu().numpy(), 1)
                
                if not loc_mode:
                  for c in range(1,5):
                    drow[c-1] += compute_tp_fn_fp(dmg_pred.data.byte().cpu().numpy(), dmg_target.data.byte().cpu().numpy(), c)
                
                running_loss += loss.item() * len(images)
            epoch_loss = running_loss / total_len

          else:
            for batch_idx, (images_pre, images_post, masks, _ , _) in enumerate(dataloaders[phase]):
              if not dice:
                masks = torch.argmax(masks,dim=1)
              masks = masks.to(device).long()
              images = torch.cat((images_pre, images_post),1)
              images = images.to(device) 
              if seperate_loss:
                loc, dmg = model(images)
                loss = criterion(loc, dmg, masks)
              else:
                output = model(images)
                loss = criterion(output, masks)

              dmg_target = masks if not dice else torch.argmax(masks,1)
              loc_target = torch.where(dmg_target > 0, torch.ones_like(dmg_target), torch.zeros_like(dmg_target))
              if output.shape[1] != 1 and len(output.shape) == 4:
                dmg_pred = torch.argmax(output,1)
                loc_mode = False
                loc_pred = torch.where(dmg_pred > 0, torch.ones_like(dmg_pred), torch.zeros_like(dmg_pred))
              else:
                loc_mode = True
                loc_pred = (1*(output > 0.5))
                if len(loc_pred.shape) == 4:
                  loc_pred = loc_pred[:,0,:,:]
                
          
              lrow += compute_tp_fn_fp(loc_pred.data.byte().cpu().numpy(), loc_target.data.byte().cpu().numpy(), 1)
              
              if not loc_mode:
                for c in range(1,5):
                  drow[c-1] += compute_tp_fn_fp(dmg_pred.data.byte().cpu().numpy(), dmg_target.data.byte().cpu().numpy(), c)
              
              running_loss += loss.item() * len(images)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)

      if phase == 'train':
        loss_dict[epoch] = (epoch_loss,)
        print('Epoch {}\nTrain Loss: {}'.format(epoch,epoch_loss).encode('ascii','ignore'))
      elif phase == 'val':
        loss_dict[epoch] += (epoch_loss,)
        locF1 = F1Recorder(lrow[0],lrow[1],lrow[2],'Buildings').f1
        if not loc_mode:
          dmgF1s = []
          for idx, key in enumerate([f'No damage     (1) ',f'Minor damage  (2) ',f'Major damage  (3) ',f'Destroyed     (4) '],1):
            dmgF1s.append(F1Recorder(drow[idx-1][0],drow[idx-1][1],drow[idx-1][2],key).f1)
        else:
          dmgF1s = [0,0,0,0]
        dmgF1 = 4/sum((x+1e-6)**-1 for x in dmgF1s)
        score = 0.3*locF1+0.7*dmgF1        
        loss_dict[epoch] += (score, locF1,dmgF1,dmgF1s[0],dmgF1s[1],dmgF1s[2],dmgF1s[3])
        state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
               'loss_dict': loss_dict
                }
        torch.save(state, os.path.join(output_path_weights,'weights_{}.pt'.format(epoch)))
        if epoch_loss <= best_val_loss:
          best_val_loss = epoch_loss
          best_epoch = epoch
        else:
          if epoch - best_epoch > lr_patience and own_sheduler:
            model.load_state_dict(torch.load(os.path.join(output_path_weights,'weights_{}.pt'.format(best_epoch)))['state_dict'])
        with open(os.path.join(output_path_loss,'loss-{}.json'.format(starttimestamp)), 'w') as fp:
          json.dump(loss_dict, fp, sort_keys=True, indent=4)
        currenttime = time.time()
        print('Validation Loss: {}\nValidation Score: {}\nLocF1: {} DmgF1: {} F1_1: {} F1_2: {} F1_3: {} F1_4: {}\n Time elapsed: {}s\nAverage Time per Epoch: {}s\nProgress: {:.0f}%\n'.format(epoch_loss,score, locF1,dmgF1,dmgF1s[0],dmgF1s[1],dmgF1s[2],dmgF1s[3],currenttime-starttime,(currenttime-starttime)/epoch,100. * epoch/num_epochs).encode('ascii','ignore'))
        if own_sheduler:
          scheduler.step(epoch_loss)#scheduler.step(score) #scheduler.step(epoch_loss)
          if epoch == 100:
            optimizer.param_groups[0]['lr'] = 0.00005
          elif epoch == 125:
            optimizer.param_groups[0]['lr'] = 0.00001
          elif epoch == 150:
            optimizer.param_groups[0]['lr'] = 0.000005
          elif epoch == 175:
            optimizer.param_groups[0]['lr'] = 0.000001
  
  model.load_state_dict(torch.load(os.path.join(output_path_weights,'weights_{}.pt'.format(best_epoch)))['state_dict'])
  return best_epoch

def test(model, dataloader, output_path_directory, seperate_loss=False, adapt = False, start_idx = 0):
  """Caller function for inference
  
  Predicts building damage maps on the images in the dataloader and outputs them to the output directory, attention, subfolders must exist already.

  Example:
    Inference of a trained TwoStream ResNet50:
    >>> model = TwoStream_Resnet50_Diff()
    >>> model.load_state_dict(torch.load(WEIGHTS)['state_dict'])
    >>> test(model, dataloader, '/output/')

  Args:
      model (torch.nn.Module): The PyTorch model to use
      dataloader (torch.utils.data.DataLoader): PyTorch dataloader with images to perform inference on
      output_path_directory (str): Path to output predicted masks to (together with ground truth masks and images) like this:
        ├── images
        │   ├── test_damage_00000_post.png
        │   ├── test_damage_00001_post.png
        │   ├── test_damage_00000_pre.png
        │   ├── test_damage_00001_pre.png
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
      seperate_loss (bool, optional): If using a model with two seperate heads for localization and damage set to True. Defaults to False.
  """  
  with torch.no_grad():
    smax = torch.nn.Softmax(dim=1)
    topil = tr.ToPILImage()
    model.eval()
    print("Saving Images to Output Path, Mode = Validation")

    if adapt:
      n = 0
      model.train()
      momenta = {}
      model.apply(lambda module: _get_momenta(module, momenta))
      for batch_idx, (images_pre, images_post, masks, indices, filenames) in tqdm(enumerate(dataloader), total = len(dataloader)):
        images = torch.cat((images_pre, images_post),1)
        images = images.to(0) 
        new_imgs = []
        for i in range(images.shape[0]):
          curr = images[i,...]
          if (1.0*(curr.sum(0) == 0)).mean() < 0.03:
            new_imgs.append(curr)
        b = len(new_imgs)
        if b > 0:
          images = torch.stack(new_imgs,0)
          images = images.to(0)
          limit = 128 if images.shape[-1] == 1024 else 512
          if len(dataloader)*images_pre.shape[0] < limit:
            momentum = b / float(n + b + (limit-len(dataloader)*images_pre.shape[0]))
          else:
            momentum = b / float(n + b)
          for module in momenta.keys():
            module.momentum = momentum
          _ = model(images)
          n += b
      model.eval()
      model.apply(lambda module: _set_momenta(module, momenta))
      if len(dataloader) > 0:
        print(len(dataloader)*images_pre.shape[0], n)

    for batch_idx, (images_pre, images_post, masks, indices, filenames) in tqdm(enumerate(dataloader), total = len(dataloader)):
      images = torch.cat((images_pre, images_post),1)
      images = images.to(0) 

      if seperate_loss:
        loc, dmg = model(images)
        loc_mask = torch.argmax(loc, dim = 1)
        output = torch.cat(((1-loc_mask).float().unsqueeze(1),dmg[:,1:,:,:] * torch.cat(4*[loc_mask.unsqueeze(1)],1)), dim = 1)
      else:
        output = model(images)
      masks = masks.to(0).long()      
      output = smax(output)
      output = torch.argmax(output,dim=1)
      #images = images.data.float().cpu().numpy()
      images = images.cpu()
      output = output.data.byte().cpu().numpy()
      masks = torch.argmax(masks, dim=1)
      masks = masks.data.byte().cpu().numpy()
      #print(masks.shape)
      #masks = np.argmax(masks[:,1:,:,:], axis=1)
      for i, index in enumerate(indices.long().cpu().numpy()):
        # HERE : SAVE ALL INDIVIDUAL IMAGES TO THEIR ORIGINAL NAMES AS PNGS
        str_index = str(index+start_idx).zfill(6)
        #pre = (np.moveaxis(images[i,:3,:,:],0,-1)*256).astype(np.uint8)
        #imwrite(output_path_directory+"images/test_damage_"+str_index+"_pre.png",pre)
        #pre = topil(images[i,:3,:,:])
        #pre.save(os.path.join(output_path_directory,"images/test_damage_"+str_index+"_pre.png"))

        #post = (np.moveaxis(images[i,3:,:,:],0,-1)*256).astype(np.uint8)
        #imwrite(output_path_directory+"images/test_damage_"+str_index+"_post.png",post)
        #post = topil(images[i,3:,:,:])
        #post.save(os.path.join(output_path_directory,"images/test_damage_"+str_index+"_post.png"))
        
        pred = np.squeeze(output[i,:,:]).astype(np.uint8)
        pred_dmg = pred.copy().astype(float)
        v = np.unique(pred_dmg)
        if True:#0 not in v:
          pred_dmg_interpolated = pred_dmg
        elif len(v) == 2:
          pred_dmg[pred_dmg == 0] = v[1]
          pred_dmg_interpolated = pred_dmg
        elif len(v) == 1:
          pred_dmg[pred_dmg == 0] = 1
          pred_dmg_interpolated = pred_dmg
        else:
          pred_dmg[pred_dmg == 0] = np.nan
          x = np.arange(0, pred_dmg.shape[1])
          y = np.arange(0, pred_dmg.shape[0])
          #mask invalid values
          pred_dmg = np.ma.masked_invalid(pred_dmg)
          xx, yy = np.meshgrid(x, y)
          #get only the valid values
          x1 = xx[~pred_dmg.mask]
          y1 = yy[~pred_dmg.mask]
          pred_dmg_new = pred_dmg[~pred_dmg.mask]

          pred_dmg_interpolated = interpolate.griddata((x1, y1), pred_dmg_new.ravel(), (xx, yy), method='nearest')
        pred_dmg_image = Image.fromarray(pred_dmg_interpolated.astype(np.uint8))
        colorize_mask_(pred_dmg_image)
        pred_dmg_image.save(os.path.join(output_path_directory,"predictions/test_damage_"+str_index+"_prediction.png"))
        pred[pred != 0] = 1
        pred_loc = Image.fromarray(pred)
        colorize_mask_(pred_loc)
        pred_loc.save(os.path.join(output_path_directory,"predictions/test_localization_"+str_index+"_prediction.png"))

        targ = np.squeeze(masks[i,:,:]).astype(np.uint8)
        targ_dmg = Image.fromarray(targ)
        colorize_mask_(targ_dmg)
        targ_dmg.save(os.path.join(output_path_directory,"targets/test_damage_"+str_index+"_target.png"))
        targ[targ != 0] = 1
        targ_loc = Image.fromarray(targ)
        colorize_mask_(targ_loc)
        targ_loc.save(os.path.join(output_path_directory,"targets/test_localization_"+str_index+"_target.png"))



def test_twostage(loc_model, dmg_model, dataloader, output_path_directory, fusion_style = 'simple', adapt = False, start_idx = 0):
  """Caller function for inference
  
  Predicts building damage maps on the images in the dataloader and outputs them to the output directory, attention, subfolders must exist already.

  Example:
    Inference of a trained TwoStream ResNet50:
    >>> loc_model = RotEqUNet()
    >>> loc_model.load_state_dict(torch.load(loc_weights)['state_dict'])
    >>> dmg_model = TwoStream_Resnet50_Diff()
    >>> dmg_model.load_state_dict(torch.load(dmg_weights)['state_dict'])
    >>> test_twostage(loc_model, dmg_model, dataloader, '/output/')

  Args:
      model (torch.nn.Module): The PyTorch model to use
      dataloader (torch.utils.data.DataLoader): PyTorch dataloader with images to perform inference on
      output_path_directory (str): Path to output predicted masks to (together with ground truth masks and images) like this:
        ├── images
        │   ├── test_damage_00000_post.png
        │   ├── test_damage_00001_post.png
        │   ├── test_damage_00000_pre.png
        │   ├── test_damage_00001_pre.png
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
      seperate_loss (bool, optional): If using a model with two seperate heads for localization and damage set to True. Defaults to False.
  """  
  with torch.no_grad():
    smax = torch.nn.Softmax(dim=1)
    smoid = torch.nn.Sigmoid()
    topil = tr.ToPILImage()
    loc_model.eval()
    dmg_model.eval()
    print("Saving Images to Output Path")

    if adapt:
      n = 0
      for batch_idx, (images_pre, images_post, masks, indices, filenames) in tqdm(enumerate(dataloader), total = len(dataloader)):
        images = torch.cat((images_pre, images_post),1)
        images = images.to(0) 
        new_imgs = []
        for i in range(images.shape[0]):
          if (images[i,...].sum(0) == 0).mean() < 0.03:
            new_imgs.append(images[i,...])
        b = len(new_imgs)
        if b > 0:
          images = torch.stack(new_imgs,0)
          images = images.to(0)
          momentum = b / float(n + b)
          for model in [loc_model, dmg_model]:
            model.train()
            momenta = {}
            model.apply(lambda module: _get_momenta(module, momenta))
            for module in momenta.keys():
                    module.momentum = momentum
            _ = model(images)
            model.eval()
            model.apply(lambda module: _set_momenta(module, momenta))
          n += b

    for batch_idx, (images_pre, images_post, masks, indices, filenames) in tqdm(enumerate(dataloader), total = len(dataloader)):
      images = torch.cat((images_pre, images_post),1)
      images = images.to(0)

      if fusion_style != 'dmg':
        loc = loc_model(images)
        loc = smoid(loc)
        loc = torch.where(loc >= 0.5, torch.ones_like(loc), torch.zeros_like(loc))
      dmg = dmg_model(images)
      dmg = smax(dmg)
      masks = masks.to(0).long()      
      #images = images.data.float().cpu().numpy()
      images = images.cpu()
      masks = torch.argmax(masks, dim=1)
      if fusion_style == 'simple':
        output = torch.cat(((1-loc).float(),(dmg * torch.cat(4*[loc],1)) + 1),1)
        output = torch.argmax(output,dim=1)
        output = output.data.byte().cpu().numpy()
      elif fusion_style == 'none':
        loc = loc.data.byte().cpu().numpy()
        if dmg.shape[1]==4:
          dmg = torch.argmax(dmg,dim=1)+1
        else:
          dmg = torch.argmax(dmg,dim=1)
        dmg = dmg.data.byte().cpu().numpy()
      elif fusion_style == 'dmg':
        loc = torch.where(masks != 0, torch.ones_like(masks), torch.zeros_like(masks))
        loc = loc.data.byte().cpu().numpy()
        if dmg.shape[1]==4:
          dmg = torch.argmax(dmg,dim=1)+1
        else:
          dmg = torch.argmax(dmg,dim=1)
        dmg = dmg.data.byte().cpu().numpy()
      elif fusion_style == 'max_freq_per_component':
        dmg = max_freq_per_component_fusion(loc,dmg)
        dmg = dmg.data.byte().cpu().numpy()
        loc = loc.data.byte().cpu().numpy()

      masks = masks.data.byte().cpu().numpy()

      #print(masks.shape)
      #masks = np.argmax(masks[:,1:,:,:], axis=1)
      for i, index in enumerate(indices.long().cpu().numpy()):
        # HERE : SAVE ALL INDIVIDUAL IMAGES TO THEIR ORIGINAL NAMES AS PNGS
        str_index = str(index+start_idx).zfill(6)
        #pre = (np.moveaxis(images[i,:3,:,:],0,-1)*256).astype(np.uint8)
        #imwrite(output_path_directory+"images/test_damage_"+str_index+"_pre.png",pre)
        pre = topil(images[i,:3,:,:])
        pre.save(os.path.join(output_path_directory,"images","test_damage_"+str_index+"_pre.png"))

        #post = (np.moveaxis(images[i,3:,:,:],0,-1)*256).astype(np.uint8)
        #imwrite(output_path_directory+"images/test_damage_"+str_index+"_post.png",post)
        post = topil(images[i,3:,:,:])
        post.save(os.path.join(output_path_directory,"images","test_damage_"+str_index+"_post.png"))

        targ = np.squeeze(masks[i,:,:]).astype(np.uint8)
        targ_dmg = Image.fromarray(targ)
        colorize_mask_(targ_dmg)
        targ_dmg.save(os.path.join(output_path_directory,"targets","test_damage_"+str_index+"_target.png"))
        targ[targ != 0] = 1
        targ_loc = Image.fromarray(targ)
        colorize_mask_(targ_loc)
        targ_loc.save(os.path.join(output_path_directory,"targets","test_localization_"+str_index+"_target.png"))
        if fusion_style in ['none', 'dmg','max_freq_per_component']:
          pred = np.squeeze(dmg[i,:,:]).astype(np.uint8)
          pred_dmg = pred.copy().astype(float)
          v = np.unique(pred_dmg)
          if 0 not in v:
            pred_dmg_interpolated = pred_dmg
          elif len(v) == 2:
            pred_dmg[pred_dmg == 0] = v[1]
            pred_dmg_interpolated = pred_dmg
          elif len(v) == 1:
            pred_dmg[pred_dmg == 0] = 1
            pred_dmg_interpolated = pred_dmg
          else:
            pred_dmg[pred_dmg == 0] = np.nan
            x = np.arange(0, pred_dmg.shape[1])
            y = np.arange(0, pred_dmg.shape[0])
            #mask invalid values
            pred_dmg = np.ma.masked_invalid(pred_dmg)
            xx, yy = np.meshgrid(x, y)
            #get only the valid values
            x1 = xx[~pred_dmg.mask]
            y1 = yy[~pred_dmg.mask]
            pred_dmg_new = pred_dmg[~pred_dmg.mask]

            pred_dmg_interpolated = interpolate.griddata((x1, y1), pred_dmg_new.ravel(), (xx, yy), method='nearest')
          pred_dmg_image = Image.fromarray(pred_dmg_interpolated.astype(np.uint8))
          pred = np.squeeze(loc[i,:,:]).astype(np.uint8)
          pred_loc = Image.fromarray(pred)
        else:
          pred = np.squeeze(output[i,:,:]).astype(np.uint8)
          pred_dmg_img = Image.fromarray(pred)
          pred[pred != 0] = 1
          pred_loc = Image.fromarray(pred)
        colorize_mask_(pred_dmg_img)
        pred_dmg_img.save(os.path.join(output_path_directory,"predictions","test_damage_"+str_index+"_prediction.png"))
        colorize_mask_(pred_loc)
        pred_loc.save(os.path.join(output_path_directory,"predictions","test_localization_"+str_index+"_prediction.png"))





  
  