import torch
from torch.utils.data import Dataset,DataLoader,random_split
import torchvision
import os
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from dataset import *
import numpy as np
from tqdm import tqdm
import numpy as np
import sys
import logging
from pathlib import Path
from utils.dice_score import *
from evaluate import evaluate
from unet import UNet
from torch import optim
import wandb
from torch.utils.data import WeightedRandomSampler
import segmentation_models_pytorch as smp
import albumentations as A
import matplotlib.pyplot as plt
from utils.logging_lr import Logger,LambdaLR
import time
import Attention_Unet_Pytorch.models.unet as unet
from patchify import patchify,unpatchify
import math

def tensor2image_cpu(tensor):
    image = 127.5*(tensor + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

transforms_img = [
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]

transforms_mask = [ 
                transforms.ToTensor()]

Tensor = torch.cuda.FloatTensor

#Directory to saved trained UNet models from train_attnunet_patches.py
model_dir = r""
model_test_folders = os.listdir(model_dir)
threshold = 0.5

for model_folder in model_test_folders:
    torch.manual_seed(42)
    
    model_base_dir = os.path.join(model_dir,model_folder)
    
    crop_size = 384
    resize_size = 512

    step_size = resize_size - crop_size
    if "S2D" in model_folder:
        dataset = DRIVE_Test_Dataset(transform_image = transforms_img, transform_mask = transforms_mask, resize_size = resize_size, clahe = False)
    elif "D2S" in model_folder:
        dataset = STARE_Test_Dataset(transform_image = transforms_img, transform_mask = transforms_mask, resize_size = resize_size, clahe = False)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    max_dice = 0
    best_epoch = 0
    for epoch in range(5,121,5):

        torch.cuda.empty_cache()
        net = unet.Unet(64, attention=True).cuda()
        if not os.path.isfile(os.path.join(model_base_dir,"UNet_epoch_%d.pth"%epoch)):
            continue
        net.load_state_dict(torch.load(os.path.join(model_base_dir,"UNet_epoch_%d.pth"%epoch)))
        dice_values = []
        for i, batch in enumerate(dataloader):

            with torch.no_grad():
                real_A = np.squeeze(batch['image'].numpy())
                mask_A = np.squeeze(batch['label'].numpy())
                name = batch['file']
                base_dir,file = os.path.split(name[0])

                patches = patchify(real_A,(3,crop_size,crop_size),step=step_size)
                patches_masks = patchify(mask_A,(crop_size,crop_size),step=step_size)
                patches_masks = np.reshape(patches_masks,(4,crop_size,crop_size))
                patches = np.reshape(patches,(4,3,crop_size,crop_size))
                patches_plot = np.transpose(patches,(0,2,3,1))
                
                patches = Tensor(patches).cuda()
                output_masks,_ = net(patches)
                output_masks = output_masks.cpu().detach()
                sig_pred = F.sigmoid(output_masks).numpy()
                sig_pred = 1.0*(sig_pred>threshold)
                sig_pred = np.squeeze(sig_pred)
                recon_image = unpatchify(np.reshape(sig_pred,(2,2,crop_size,crop_size)),(resize_size,resize_size))
                dice_values.append(dice_coeff(torch.Tensor(recon_image),torch.Tensor(mask_A)).item())
            
        dice_values_np = np.asarray(dice_values)
        dice_max = np.amax(dice_values_np)
        dice_min = np.amin(dice_values_np)
        dice_mean = np.mean(dice_values_np)
        dice_std = np.std(dice_values_np)
        dice_err = dice_std/(math.sqrt(len(dataloader)))

        if dice_mean > max_dice:
            best_epoch = epoch
            max_dice = dice_mean
            max_dice_err = dice_err
    print("For model %s: Mean DSC %f, Error %f."%(model_folder,max_dice,max_dice_err))


                                                


                        
            
        
        
    
