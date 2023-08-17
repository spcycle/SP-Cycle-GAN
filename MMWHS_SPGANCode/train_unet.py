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
from dice_score import dice_loss
from torch import optim
import wandb
from torch.utils.data import WeightedRandomSampler
import segmentation_models_pytorch as smp
import albumentations as A
import matplotlib.pyplot as plt
from utils import LambdaLR
from utils import Logger
import time
import math
import random
##Set original_seg_dir to the directory of translated images generated by "translate_images_fromGANs.py"
##Set original_dir_checkpoint to desired output directory for unet models.
original_seg_dir = r""
original_dir_checkpoint = r"D:\Saved_Models\MMWHS_UNet"

folders = os.listdir(original_seg_dir)
for file in folders:
    torch.manual_seed(42)
    random.seed(42)
    torch.cuda.empty_cache()
    seg_dir = os.path.join(os.path.join(original_seg_dir,file),"Images")
    dir_checkpoint = os.path.join(original_dir_checkpoint,file)
    if not os.path.isdir(dir_checkpoint):
        os.mkdir(dir_checkpoint)

    transforms_img = [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5))]
    clahe = False
    if "clahe" in file:
        clahe = True
    if "MR_to_CT" in file:
        mode = "MRtoCT"
    elif "CT_to_MR" in file:
        mode = "CTtoMR"

    print("Loading patches and masks")    
    dataset = SegGAN_MMWHS_UNet_Train(image_dir = seg_dir, transform_image = transforms_img, transform_mask = None, clahe = clahe, mode = mode)
    print(file,clahe)
    batch_size = 8

    tversky = smp.losses.TverskyLoss(mode = 'multiclass',classes=3,eps=1e-07,alpha = 0.3, beta = 0.7,smooth=0.0)    
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


##    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
##    logging.info(f'Using device {device}')


    ##net.to(device=device)
    net = smp.UnetPlusPlus(encoder_weights = None,decoder_attention_type = 'scse',in_channels=1,classes=3).cuda()

    learning_rate = 0.001
    epochs = 80
    amp = True


    ##optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(),lr=learning_rate)
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(epochs, 0, epochs/2).step)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    global_step = 0
    logger = Logger(epochs, len(train_dataloader))

    net.train()
    for epoch in range(0, epochs):
        for i, batch in enumerate(train_dataloader):
                optimizer.zero_grad()

        
                image = batch['A_img'].to(device='cuda')
                mask_true = batch['A_mask'].to(device='cuda').long()
                pred_mask = net(image)
                loss = torch.pow(tversky(pred_mask,mask_true),3/4)
                loss.backward()
                optimizer.step()
                logger.log(losses={'loss_G': loss}, 
                    images={'image': image[0,:,:,:],'pred_mask': F.softmax(pred_mask[0,:,:,:]),'true_mask': mask_true[0,:,:]/2},epoch=epoch,draw_images=True)
##            
        lr_sched.step()
        if (epoch+1)%5 == 0:
            torch.save(net.state_dict(), os.path.join(dir_checkpoint,('UNet_epoch_%s.pth'%str(epoch+1))))

