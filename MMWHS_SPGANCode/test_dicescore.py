import torch
from torch.utils.data import Dataset,DataLoader,random_split
import torchvision
import os
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from datasets import *
import numpy as np
from tqdm import tqdm
import numpy as np
import sys
import logging
from pathlib import Path
from dice_score import *
from torch import optim
import wandb
from torch.utils.data import WeightedRandomSampler
import segmentation_models_pytorch as smp
import albumentations as A
import matplotlib.pyplot as plt
from utils import Logger,LambdaLR
import time
from patchify import patchify,unpatchify
import math
import random


def tensor2image_cpu(tensor):
    image = 127.5*(tensor + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

transforms_img = [
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))]

Tensor = torch.cuda.FloatTensor
##Set to saved directory of trained UNet models on the translated patches
## name the MR to CT baseline model: "untranslated_mr" and the CT to MR baseline model
##"untranslated_ct".

model_dir = r""
model_test_folders = os.listdir(model_dir)
num = 0

for model_folder in model_test_folders:
    torch.manual_seed(42)
    random.seed(42)

    model_base_dir = os.path.join(model_dir,model_folder)
    torch.cuda.empty_cache()
    net = smp.UnetPlusPlus(encoder_weights = None,decoder_attention_type = 'scse',in_channels=1,classes=3).cuda()

    if "MR_to_CT" in model_folder:
        mode = "MRtoCT"
    elif "CT_to_MR" in model_folder:
        mode = "CTtoMR"
    elif "untranslated_mr" in model_folder:
        mode = "MRtoCT"
    elif "untranslated_ct" in model_folder:
        mode = "CTtoMR"

    dataset = SegGAN_MMWHS_Test_CT_or_MR(transform_image = transforms_img, transform_mask = None, mode = mode)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    max_dice = 0
    best_epoch = 0

    for epoch in range(5,81,5):

        net.load_state_dict(torch.load(os.path.join(model_base_dir,"UNet_epoch_%d.pth"%epoch)))

        dice_values_LV = []
        dice_values_MYO = []
        dice_values = []
        dice_values_BACK = []
        
        for i, batch in enumerate(dataloader):

            with torch.no_grad():
                image = batch['A_img'].to(device='cuda')
                mask_true = batch['A_mask'].to(device='cuda').long()

                mask_true = torch.permute(torch.nn.functional.one_hot(mask_true, num_classes= 3),(0,3,1,2))
                pred_mask = net(image)
                pred_mask = torch.round(torch.nn.functional.softmax(pred_mask,dim=1))                
                dice_values.append(multiclass_dice_coeff(pred_mask,mask_true.float()).item())
                dice_values_BACK.append(dice_coeff(pred_mask[0,0,:,:],mask_true[0,0,:,:].float()).item())
                dice_values_LV.append(dice_coeff(pred_mask[0,1,:,:],mask_true[0,1,:,:].float()).item())
                dice_values_MYO.append(dice_coeff(pred_mask[0,2,:,:],mask_true[0,2,:,:].float()).item())
            
        dice_values_np = np.asarray(dice_values)
        dice_max = np.amax(dice_values_np)
        dice_min = np.amin(dice_values_np)
        dice_mean = np.mean(dice_values_np)
        dice_std = np.std(dice_values_np)
        dice_err = dice_std/(math.sqrt(len(dataloader)))

        dice_values_LV_np = np.asarray(dice_values_LV)
        dice_max_LV = np.amax(dice_values_LV_np)
        dice_min_LV = np.amin(dice_values_LV_np)
        dice_mean_LV = np.mean(dice_values_LV_np)
        dice_std_LV = np.std(dice_values_LV_np)
        dice_err_LV = dice_std_LV/(math.sqrt(len(dataloader)))

        dice_values_MYO_np = np.asarray(dice_values_MYO)
        dice_max_MYO = np.amax(dice_values_MYO_np)
        dice_min_MYO = np.amin(dice_values_MYO_np)
        dice_mean_MYO = np.mean(dice_values_MYO_np)
        dice_std_MYO = np.std(dice_values_MYO_np)
        dice_err_MYO = dice_std_MYO/(math.sqrt(len(dataloader)))

        dice_values_BACK_np = np.asarray(dice_values_BACK)
        dice_max_BACK = np.amax(dice_values_BACK_np)
        dice_min_BACK = np.amin(dice_values_BACK_np)
        dice_mean_BACK = np.mean(dice_values_BACK_np)
        dice_std_BACK = np.std(dice_values_BACK_np)
        dice_err_BACK = dice_std_BACK/(math.sqrt(len(dataloader)))

        if dice_mean > max_dice:
            best_epoch = epoch
            max_dice = dice_mean
            max_dice_err = dice_err
            max_dice_LV = dice_mean_LV
            max_err_LV = dice_err_LV
            max_dice_MYO = dice_mean_MYO
            max_err_MYO = dice_err_MYO
            max_dice_BACK = dice_mean_MYO
            max_err_BACK = dice_err_BACK
            

    print("For model %s: Mean DSC %f Error: %f. LV DSC %f Error %f. MYO DSC %f Error %f BACK DSC %f Error %f "%(model_folder,max_dice,max_dice_err,
                                                                                                                max_dice_LV,max_err_LV,max_dice_MYO,max_err_MYO,max_dice_BACK,max_err_BACK))
            
                                                


                        
            
        
        
    
