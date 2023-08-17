#!/usr/bin/python3

import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
from dice_score import dice_loss
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets import *
import segmentation_models_pytorch as smp
import os
import math
import random

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training') 
parser.add_argument('--batchSize', type=int, default=8, help='size of the batches')
##parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate') #0.0002
parser.add_argument('--decay_epoch', type=int, default=50, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=192, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--name',type=str)
##parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)

torch.manual_seed(42)
random.seed(42)
##This code will automatically train SP Cycle-GAN or Default Cycle-GAN models for MM-WHS CT to MR or MR to CT DA,
##and can train multiple models in one run of the code. Model parameters are simply specified by the name of the model specified
##in the "names" list below. To specify what DA Direction to use: Include "MR_to_CT" or "CT_to_MR" in the model name. To use the SP-CycleGAN loss function: Include "seggan" in the name, along with "Gamma5","Gamma4", "Gamma3" to specify the
## hyperparameter value zeta in the SP Cycle loss from the paper. To train a standard Cycle-GAN: only use the term "Default" in the name or simply do not
## include "seggan" or "gamma". Eg: To train a MR to CT SP-Cycle GAN model with zeta = 3.0, name the model "MR_to_CT_seggan_gamma3". 
## If you want to see Visdom log and output, run py -m visdom.server to run a vidsom instance and then run this code. You can then navigate to the
## specified url and see GAN image training process, and loss logs.

##NOTE: Make sure that the CT_MR_2D_Dataset_DA-master folder with the mm-whs data is in the directory of this code to ensure function. 

names = ["MR_to_CT_seggan_gamma3"]

for name in names:
    torch.manual_seed(42)
    random.seed(42)

    ##set base directory for saved GAN models to your choosing
    base_dir = r"GAN_TrainedModels"
    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)

    save_dir = os.path.join(base_dir, name)
    torch.cuda.empty_cache()

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    ###### Definition of variables ######
    # Networks
    netG_A2B = Generator(opt.input_nc, opt.output_nc).cuda()
    netG_B2A = Generator(opt.output_nc, opt.input_nc).cuda()
    netD_A = Discriminator(opt.input_nc).cuda()
    netD_B = Discriminator(opt.output_nc).cuda()


    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)
    tversky = smp.losses.TverskyLoss(mode = 'multiclass', classes=3, eps=1e-07 ,alpha = 0.3, beta = 0.7,smooth=0.0)

    # Lossess
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                    lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
    label_A = Tensor(opt.batchSize, opt.size,opt.size)
    label_B = Tensor(opt.batchSize, opt.size,opt.size)

    target_real = Variable(Tensor(opt.batchSize,1).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batchSize,1).fill_(0.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Dataset loader
    transforms_image = [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5))]

    transforms_mask = None

    ##dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True), 
    ##                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)
    if "CT_to_MR" in name:
        mode = "CTtoMR"
    elif "MR_to_CT" in name:
        mode = "MRtoCT"

    dataset = SegGAN_Dataset_MMWHS(transform_image=transforms_image,crop_size = opt.size, mode = mode)
    dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True)

    # Loss plot
    logger = Logger(opt.n_epochs, len(dataloader))

    #Unet setup

    if "seggan" in name:
        unet_model_a_rec = smp.UnetPlusPlus(encoder_weights = None,decoder_attention_type = 'scse',in_channels=1,classes=3).cuda()
        criterion_unet_a = nn.CrossEntropyLoss()

        lr_unet_a = 0.001
        ##
        optimizer_unet_a_rec = torch.optim.Adam(unet_model_a_rec.parameters(), lr=lr_unet_a, betas=(0.5, 0.999))
        lr_scheduler_unet_a_rec = torch.optim.lr_scheduler.LambdaLR(optimizer_unet_a_rec, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    ###################################
    torch.autograd.set_detect_anomaly(True)
    
    criterion = nn.BCEWithLogitsLoss()
    if "gamma1" in name:
        gamma = 1.0
    elif "gamma2" in name:
        gamma = 2.0
    elif "gamma3" in name:
        gamma = 3.0
    elif "gamma4" in name:
        gamma = 4.0
    elif "gamma5" in name:
        gamma = 5.0
    ###### Training ######
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
            # Set model input
            
    ##        optimizer_G.zero_grad()
            optimizer_G.zero_grad()
            
            real_A = Variable(input_A.copy_(batch['A_img']))
            real_B = Variable(input_B.copy_(batch['B_img']))
            
            mask_A = Variable(label_A.copy_( batch['A_mask'])).long()
            mask_B = Variable(label_B.copy_( batch['B_mask']))
            
            ###### Generators A2B and B2A ######

            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            fake_B = netG_A2B(real_A)
            fake_A = netG_B2A(real_B)
            
            same_B = netG_A2B(real_B)
            same_A = netG_B2A(real_A)
            
            recovered_A = netG_B2A(fake_B)
            recovered_B = netG_A2B(fake_A)
    ##
            if "seggan" in name:
                pred_mask_a_1 = unet_model_a_rec(recovered_A)
                loss_unet_a_1 = gamma*torch.pow(tversky(pred_mask_a_1,mask_A),3/4)
                loss_unet_a_1.backward(retain_graph=True)        

            loss_identity_B = criterion_identity(same_B, real_B)*5.0
            loss_identity_A = criterion_identity(same_A, real_A)*5.0
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

            loss_G.backward()
            optimizer_G.step()
            
            if "seggan" in name:
            
                optimizer_unet_a_rec.zero_grad()
                pred_mask_a_1_v2 = unet_model_a_rec(recovered_A.detach())
                loss_unet_a_1_v2 = torch.pow(tversky(pred_mask_a_1_v2,mask_A),3/4)
                loss_unet_a_1_v2.backward()
                optimizer_unet_a_rec.step()


            ###### Discriminator A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()

            optimizer_D_A.step()
            ###################################

            ###### Discriminator B ######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)
            
            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()

            optimizer_D_B.step()
            
            if "seggan" in name:
                logger.log(losses={'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),'loss_SEGGAN':(loss_unet_a_1),
                            'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)})
            else:
                logger.log(losses={'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                            'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)})


        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
        if "seggan" in name:
            lr_scheduler_unet_a_rec.step()
        
        # Save models checkpoints
        if (epoch+1)%10 == 0:
            torch.save(netG_A2B.state_dict(), os.path.join(save_dir,('netG_A2B_epoch_%s.pth'%str(epoch+1))))
            torch.save(netG_B2A.state_dict(), os.path.join(save_dir,('netG_B2A_epoch_%s.pth'%str(epoch+1))))
            if "seggan" in name:
                torch.save(unet_model_a_rec.state_dict(), os.path.join(save_dir,('UnetA_rec_epoch_%s.pth'%str(epoch+1))))
            
        torch.save(netG_A2B.state_dict(), os.path.join(save_dir,('netG_A2B_current.pth')))
        torch.save(netG_B2A.state_dict(), os.path.join(save_dir,('netG_B2A_current.pth')))
        if "seggan" in name:
            torch.save(unet_model_a_rec.state_dict(), os.path.join(save_dir,('UnetA_rec_current.pth')))

###################################
