import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
import matplotlib.pyplot as plt
from dice_score import dice_loss
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import time
from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets import *
import segmentation_models_pytorch as smp
import os
import numpy as np
import albumentations as A
from patchify import patchify,unpatchify
import math
import random

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)
def tensor2image_cpu(tensor):
    image = 127.5*(tensor + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=6000, help='number of epochs of training') 
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
##parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate') #0.0002
parser.add_argument('--decay_epoch', type=int, default=3000, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=384, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
##parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)

##This code will translate the source domain training set patches using the trained GAN models from
##train_SPCycleGAN.py. Run this code to translate the patches, and then you can train the segmentation models using
## the code in the Train_Segmentation Models folder.

## Specify directory of saved GAN models.
original_model_dir = r""
folders = os.listdir(model_dir)
for folder in folders:
    torch.manual_seed(42)
    random.seed(42)
    original_size = 512
    crop_size = 384
    transforms_image = [transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]

    transforms_mask = [transforms.ToTensor()]

    #be sure to update the below folder directories to the provided STARE/DRIVE folders with corresponding names. 
    
    if "S2D" in folder:
        dataset = SegGAN_STARE_Patches_To_DRIVE_With_Label(a_dir = r"STARE_Labelled",b_dir=r"DRIVE\training",
                                            transform_image=transforms_image,transform_label=transforms_mask)
    elif "D2S" in folder:
        dataset = SegGAN_DRIVE_Patches_To_STARE_With_Label_noBMask(a_dir = r"DRIVE\training",b_dir=r"STARE_Unlabelled",
                                            transform_image=transforms_image,transform_label=transforms_mask)
        
    model_dir = os.path.join(original_model_dir,folder)

    name = folder.replace("saved_checkpoints_","")

    #Set output path for translated image patches and labels.
    
    output_path = r""
    original_size = 512
    crop_size = 384
    step_size = original_size - crop_size

    
    train_set,val_set = torch.utils.data.random_split(dataset, [math.floor(0.8*len(dataset)),math.floor(0.2*len(dataset))])
    dataloader = DataLoader(train_set, batch_size=1, shuffle=True)

    for epoch in range(10,201,10):
        
        torch.cuda.empty_cache()
        netG_A2B_C = Generator(opt.input_nc, opt.output_nc).cuda()
        netG_A2B_C.load_state_dict(torch.load(os.path.join(model_dir,"netG_A2B_epoch_%d.pth"%(epoch))))

        Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
        input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
        input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
        label_A = Tensor(1,1,512,512)
        label_B = Tensor(1,1,512,512)

        if torch.cuda.is_available() and not opt.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

        logger = Logger(opt.n_epochs, len(dataloader))

        target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
        target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)
        fake_A_buffer = ReplayBuffer()
        fake_B_buffer = ReplayBuffer()

        time = 0.6
        mode = "save"
        name_dir = "%s_epoch_%d"%(name,epoch)
        print(name_dir)
        output_path_img  = os.path.join(output_path,name_dir)
        if not os.path.isdir(output_path_img):
            os.mkdir(output_path_img)
        for i, batch in enumerate(dataloader):

            real_A = np.squeeze(batch['A_img'].numpy())
            mask_A = np.squeeze(batch['A_mask'].numpy())
            name_img = batch['file']
            base_dir_img,file_img = os.path.split(name_img[0])
            print(file_img)
            print(real_A.shape)

            patches = patchify(real_A,(3,crop_size,crop_size),step=step_size)
            patches_mask = patchify(mask_A,(crop_size,crop_size),step=step_size)

            patches = np.reshape(patches,(4,3,crop_size,crop_size))
            patches_mask = np.reshape(patches_mask,(4,crop_size,crop_size))
            patches = Tensor(patches).cuda()
            outpatches = netG_A2B_C(patches)
            
            outpatches = np.transpose(outpatches.cpu().detach().numpy(),(0,2,3,1))
            image_base = os.path.join(output_path_img,"Images")
            if not os.path.isdir(image_base):
                os.mkdir(image_base)

            label_base = os.path.join(output_path_img,"Label")
            if not os.path.isdir(label_base):
                os.mkdir(label_base)

            for i in range(0,4):
                new_file = file_img[:-4] + "_patch%d"%(i+1)
                image = Image.fromarray(tensor2image_cpu(outpatches[i,:,:,:]))
                image.save(os.path.join(image_base,new_file + '.jpg'))
                mask = Image.fromarray(255*(patches_mask[i,:,:]).astype(np.uint8))
                mask.save(os.path.join(label_base,new_file + '.ppm'))
            
            
               

           
