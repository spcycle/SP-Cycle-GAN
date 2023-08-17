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
parser.add_argument('--size', type=int, default=192, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
##parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()

##Set to the GAN Model directory
folders = os.listdir(r"GAN_TrainedModels")
for folder in folders:
    torch.cuda.empty_cache()
    for epoch in range(10,101,10):
        torch.manual_seed(42)
        random.seed(42)
        netG_A2B_C = Generator(opt.input_nc, opt.output_nc).cuda()
        netG_A2B_C.load_state_dict(torch.load(r"D:\Saved_Models\GAN_TrainedModels\%s\netG_A2B_epoch_%d.pth"%(folder,epoch)))

        Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor

        if torch.cuda.is_available() and not opt.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        if "CT_to_MR" in folder:
            mode = "CTtoMR"
        elif "MR_to_CT" in folder:
            mode = "MRtoCT"

        transforms_image = [transforms.ToTensor(),
                        transforms.Normalize((0.5), (0.5))]

        #set output path of translated data
        output_path = r"MMWHS_TranslatedData"
        dataset = SegGAN_Dataset_MMWHS(transform_image=transforms_image, mode=mode)
        dataloader = DataLoader(dataset, batch_size= 1, shuffle=True)


        logger = Logger(opt.n_epochs, len(dataloader))

        time = 0.6
        mode = "save"
        name_dir = "%s_%d"%(folder,epoch)
        output_path  = os.path.join(output_path,name_dir)
        if mode == "visualize":
            f, (ax1,ax2,ax3) = plt.subplots(1, 3, sharey=False)
            f.show()
        if mode == "save":
            if not os.path.isdir(output_path):
                os.mkdir(output_path)
        for i, batch in enumerate(dataloader):
        ##    real_A = Variable(input_A.copy_(batch['A_img'])).cuda()
            real_A = batch['A_img'].numpy()
            mask_A_GS = (255*(np.squeeze(batch['A_mask'].numpy())/2)).astype(np.uint8)
            mask_A = np.squeeze(torch.nn.functional.one_hot(batch['A_mask'].long(),num_classes=3).numpy())
            name = batch['file']
            base_dir,file = os.path.split(name[0])
            whole_img_translated = Tensor(real_A).cuda()
            whole_img_translated = np.squeeze(netG_A2B_C(whole_img_translated).cpu().detach().numpy())

            im_PIL = Image.fromarray(tensor2image_cpu(whole_img_translated))
            mask_PIL = Image.fromarray(255*mask_A.astype(np.uint8))
            trans_mask = Image.new('LA',im_PIL.size,(0,200))
            composite = Image.composite(im_PIL,mask_PIL,trans_mask).convert('L')
            composite = np.asarray(composite)
            
            if mode == "visualize":
                if i == 0:       
                    im_1 = ax1.imshow(tensor2image_cpu(whole_img_translated),cmap='gray',interpolation='none')
                    im_3 = ax3.imshow(composite,interpolation='none',cmap='gray')
                    im_2 = ax2.imshow(tensor2image_cpu(np.squeeze(real_A)),cmap='gray',interpolation='none')
                else:
                    im_1.set_data(tensor2image_cpu(whole_img_translated))
                    im_2.set_data(tensor2image_cpu(np.squeeze(real_A)))
                    im_3.set_data(composite)
                plt.pause(time)
            

            elif mode == "save":
                image_base = os.path.join(output_path,"Images")
                if not os.path.isdir(image_base):
                    os.mkdir(image_base)

                new_file = file[:-7]    
                image = Image.fromarray(tensor2image_cpu(whole_img_translated))
                image.save(os.path.join(image_base,new_file + '.jpg'))
                
            

       

           
