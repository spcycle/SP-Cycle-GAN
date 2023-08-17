import os
import numpy as np
import torch
from torchvision.io import read_image
from os import listdir
from os.path import splitext
from pathlib import Path
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import glob
from torchvision import transforms
import cv2
import albumentations as A


def ApplyClaheColor(image):
    
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    return img_output

def Denoise(image):

    image = cv2.bilateralFilter(image, d=5, sigmaColor=9, sigmaSpace=9)
    return image


class SegGAN_Dataset(Dataset):
    def __init__(self, a_dir, transform_image=None,transform_mask=None, clahe = False, mode='train'):
        self.transform_im = transforms.Compose(transform_image)
        self.transform_label = transforms.Compose(transform_mask)
        self.clahe = clahe
        self.flipper = A.Compose(
            transforms = [
                A.VerticalFlip(p=0.5)
                ]
            )
        
##        
        self.images_A = sorted(glob.glob(os.path.join(a_dir,'Images') + '/*.*'))
        self.labels_A = sorted(glob.glob(os.path.join(a_dir,'Label') + '/*.*'))        

    def __getitem__(self, index):

        image_a = np.asarray(Image.open(self.images_A[index % len(self.images_A)]))
        if self.clahe == True:
            image_a = cv2.cvtColor(image_a,cv2.COLOR_RGB2BGR)
            image_a = Denoise(ApplyClaheColor(image_a))

        mask_a = np.asarray(Image.open(self.labels_A[index % len(self.labels_A)]))
        flipped = self.flipper(image=image_a,mask=mask_a)
        image_a = flipped['image']
        mask_a = flipped['mask']

        image_a = self.transform_im(Image.fromarray(image_a))
        mask_a = self.transform_label(Image.fromarray(mask_a))


        return {'image': image_a, 'label': mask_a}

    def __len__(self):
        return len(self.images_A)

class STARE_Baseline(Dataset):
    def __init__(self, a_dir = r"STARE_Labelled", transform_image=None,transform_mask=None, crop_size = None, resize_size = None, clahe = False, mode='train'):
        self.transform_im = transforms.Compose(transform_image)
        self.transform_label = transforms.Compose(transform_mask)
        self.clahe = clahe
        self.flipper = A.Compose(
            transforms = [
                A.Resize(height = resize_size, width = resize_size),
                A.RandomCrop(height = crop_size, width = crop_size),
                A.VerticalFlip(p=0.5)
                ]
            )
        
##        self.unaligned = unaligned
##        
        self.images_A = sorted(glob.glob(os.path.join(a_dir,'Images') + '/*.*'))
        self.labels_A = sorted(glob.glob(os.path.join(a_dir,'Label') + '/*.*'))        

    def __getitem__(self, index):
##        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        image_a = np.asarray(Image.open(self.images_A[index % len(self.images_A)]))
        if self.clahe == True:
            image_a = cv2.cvtColor(image_a,cv2.COLOR_RGB2BGR)
            image_a = Denoise(ApplyClaheColor(image_a))

        mask_a = np.asarray(Image.open(self.labels_A[index % len(self.labels_A)]))
        flipped = self.flipper(image=image_a,mask=mask_a)
        image_a = flipped['image']
        mask_a = flipped['mask']

        image_a = self.transform_im(Image.fromarray(image_a))
        mask_a = self.transform_label(Image.fromarray(mask_a))


        return {'image': image_a, 'label': mask_a}

    def __len__(self):
        return len(self.images_A)
    
class DRIVE_Baseline(Dataset):
    def __init__(self, a_dir = r"DRIVE\training", transform_image=None,transform_mask=None, crop_size = None, resize_size = None, clahe = False, mode='train'):
        self.transform_im = transforms.Compose(transform_image)
        self.transform_label = transforms.Compose(transform_mask)
        self.clahe = clahe
        self.flipper = A.Compose(
            transforms = [
                A.Resize(height = resize_size, width = resize_size),
                A.RandomCrop(height = crop_size, width = crop_size),
                A.VerticalFlip(p=0.5)
                ]
            )
        
##        self.unaligned = unaligned
##        
        self.images_A = sorted(glob.glob(os.path.join(a_dir,'Images') + '/*.*'))
        self.labels_A = sorted(glob.glob(os.path.join(a_dir,'Label') + '/*.*'))        

    def __getitem__(self, index):
##        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        image_a = np.asarray(Image.open(self.images_A[index % len(self.images_A)]))
        if self.clahe == True:
            image_a = cv2.cvtColor(image_a,cv2.COLOR_RGB2BGR)
            image_a = Denoise(ApplyClaheColor(image_a))

        mask_a = np.asarray(Image.open(self.labels_A[index % len(self.labels_A)]))
        flipped = self.flipper(image=image_a,mask=mask_a)
        image_a = flipped['image']
        mask_a = flipped['mask']

        image_a = self.transform_im(Image.fromarray(image_a))
        mask_a = self.transform_label(Image.fromarray(mask_a))


        return {'image': image_a, 'label': mask_a}

    def __len__(self):
        return len(self.images_A)

class DRIVE_Test_Dataset(Dataset):
    def __init__(self, a_dir = r"DRIVE_Test_Label", transform_image=None,transform_mask=None,
                 resize_size = None, clahe = False, mode='train'):
        self.clahe = clahe
        self.transform_im = transforms.Compose(transform_image)
        self.transform_label = transforms.Compose(transform_mask)
        self.flipper = A.Compose(
            transforms = [
                A.Resize(width = resize_size, height= resize_size)
                ]
            )

        self.images_A = sorted(glob.glob(os.path.join(a_dir,'images') + '/*.*'))
        self.labels_A = sorted(glob.glob(os.path.join(a_dir,'1st_manual') + '/*.*'))        

    def __getitem__(self, index):

        image_a = np.asarray(Image.open(self.images_A[index % len(self.images_A)]))
        file = self.images_A[index % len(self.images_A)]
        if self.clahe == True:
            image_a = cv2.cvtColor(image_a,cv2.COLOR_RGB2BGR)
            image_a = Denoise(ApplyClaheColor(image_a))

        mask_a = np.asarray(Image.open(self.labels_A[index % len(self.labels_A)]))
        flipped = self.flipper(image=image_a,mask=mask_a)
        image_a = flipped['image']
        mask_a = flipped['mask']

        image_a = self.transform_im(Image.fromarray(image_a))
        mask_a = self.transform_label(Image.fromarray(mask_a))


        return {'image': image_a, 'label': mask_a, "file": file}

    def __len__(self):
        return len(self.images_A)
class STARE_Test_Dataset(Dataset):
    def __init__(self, a_dir = r"STARE_Labelled", transform_image=None,transform_mask=None,
                 resize_size = None, clahe = False, mode='train'):
        self.clahe = clahe
        self.transform_im = transforms.Compose(transform_image)
        self.transform_label = transforms.Compose(transform_mask)
        self.flipper = A.Compose(
            transforms = [
                A.Resize(width = resize_size, height= resize_size)
                ]
            )

        self.images_A = sorted(glob.glob(os.path.join(a_dir,'Images') + '/*.*'))
        self.labels_A = sorted(glob.glob(os.path.join(a_dir,'Label') + '/*.*'))        

    def __getitem__(self, index):
##        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        image_a = np.asarray(Image.open(self.images_A[index % len(self.images_A)]))
        file = self.images_A[index % len(self.images_A)]
        if self.clahe == True:
            image_a = cv2.cvtColor(image_a,cv2.COLOR_RGB2BGR)
            image_a = Denoise(ApplyClaheColor(image_a))

        mask_a = np.asarray(Image.open(self.labels_A[index % len(self.labels_A)]))
        flipped = self.flipper(image=image_a,mask=mask_a)
        image_a = flipped['image']
        mask_a = flipped['mask']

        image_a = self.transform_im(Image.fromarray(image_a))
        mask_a = self.transform_label(Image.fromarray(mask_a))


        return {'image': image_a, 'label': mask_a, "file": file}

    def __len__(self):
        return len(self.images_A)
   
class SegGAN_Dataset_Generate_Masks(Dataset):
    def __init__(self, a_dir, transform_image=None):
        self.transform_im = transforms.Compose(transform_image)
        self.images_A = sorted(glob.glob(a_dir + '/*.*'))
    def __getitem__(self, index):
##        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        image_a = np.asarray(Image.open(self.images_A[index % len(self.images_A)]))
        image_a = cv2.cvtColor(image_a,cv2.COLOR_RGB2BGR)
        image_a = Denoise(ApplyClaheColor(image_a))

        image_a = self.transform_im(Image.fromarray(image_a))
        file_name = self.images_A[index % len(self.images_A)]

        return {'image': image_a, 'file_name': file_name}

    def __len__(self):
        return len(self.images_A)
