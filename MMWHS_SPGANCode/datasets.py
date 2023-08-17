import glob
import random
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import albumentations as A
import numpy as np
import cv2
import nibabel as nib

def ApplyClaheColor(image):
    
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    return img_output

def Denoise(image):

    image = cv2.bilateralFilter(image, d=5, sigmaColor=9, sigmaSpace=9)
    return image

class SegGAN_Dataset_MMWHS(Dataset):
    def __init__(self, transform_image=None,transform_mask=None, crop_size = 192, mode='CTtoMR', clahe = False):
        
        torch.manual_seed(42)
        random.seed(42)
        self.transform_im = transforms.Compose(transform_image)
        self.transform_label = transforms.Compose(transform_mask)
        self.clahe = clahe
        self.flipper = A.Compose(
            transforms = [
                A.CenterCrop(width=crop_size, height=crop_size)
                ]
            )
        
##        self.unaligned = unaligned
##
        if mode == "CTtoMR":
            a_dir = r"CT_MR_2D_Dataset_DA-master\CT_withGT"
            b_dir = r"CT_MR_2D_Dataset_DA-master\MR_woGT_Comb"
        elif mode == "MRtoCT":
            a_dir = r"CT_MR_2D_Dataset_DA-master\MR_withGT"
            b_dir = r"CT_MR_2D_Dataset_DA-master\CT_woGT"
            
        self.images_A = sorted(glob.glob(os.path.join(a_dir,'Images') + '/*.*'))
        self.labels_A = sorted(glob.glob(os.path.join(a_dir,'Label') + '/*.*'))
        self.images_B = sorted(glob.glob(os.path.join(b_dir,'Images') + '/*.*'))
        self.labels_B = sorted(glob.glob(os.path.join(b_dir,'Label') + '/*.*'))
        

    def __getitem__(self, index):
##        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        image_a = nib.load(self.images_A[index % len(self.images_A)])
        name = self.images_A[index % len(self.images_A)]

        mask_load_a = nib.load(self.labels_A[index % len(self.images_A)])
        image_a = np.squeeze(image_a.get_fdata())
        image_a = ((image_a - image_a.min()) * 255. / (image_a.max() - image_a.min())).astype(np.uint8)
        mask_load_a = np.squeeze(mask_load_a.get_fdata())
        x,y = mask_load_a.shape
        mask_a = np.zeros((x,y))
        mask_a[mask_load_a == 500] = 1
        mask_a[mask_load_a == 205] = 2

        if self.clahe == True:
            image_a = cv2.cvtColor(image_a,cv2.COLOR_RGB2BGR)
            image_a = Denoise(ApplyClaheColor(image_a))
        
        flipped = self.flipper(image=image_a,mask=mask_a)
        image_a = flipped['image']
        mask_a = flipped['mask']

        rand = random.randint(0, len(self.images_B) - 1)
        
        image_b = nib.load(self.images_B[rand])
        mask_load_b = nib.load(self.labels_B[rand])

        image_b = np.squeeze(image_b.get_fdata())
        image_b = ((image_b - image_b.min()) * 255. / (image_b.max() - image_b.min())).astype(np.uint8)
        
        mask_load_b = np.squeeze(mask_load_b.get_fdata())
        x,y = mask_load_b.shape
        mask_b = np.zeros((x,y))
        mask_b[mask_load_b == 500] = 1
        mask_b[mask_load_b == 205] = 2

        if self.clahe == True:
            image_b = cv2.cvtColor(image_b,cv2.COLOR_RGB2BGR)
            image_b = Denoise(ApplyClaheColor(image_b))
        
        flipped = self.flipper(image=image_b,mask=mask_b)
        image_b = flipped['image']
        mask_b = flipped['mask']
        
        image_a = self.transform_im(Image.fromarray(image_a))
        mask_a = torch.from_numpy(mask_a)

        image_b = self.transform_im(Image.fromarray(image_b))
        mask_b = torch.from_numpy(mask_b)
        
        return {'A_img': image_a, 'A_mask': mask_a, 'B_img': image_b, 'B_mask': mask_b, 'file':name}

    def __len__(self):
        return len(self.images_A)

class SegGAN_MMWHS_Translate(Dataset):
    def __init__(self, a_dir = r"CT_MR_2D_Dataset_DA-master\CT_withGT", b_dir=r"CT_MR_2D_Dataset_DA-master\MR_withGT",
                 transform_image=None,transform_mask=None, crop_size = 192, mode='CTtoMR', clahe = False):
        torch.manual_seed(42)
        self.transform_im = transforms.Compose(transform_image)
        self.transform_label = transforms.Compose(transform_mask)
        self.clahe = clahe
        self.flipper = A.Compose(
            transforms = [
                A.CenterCrop(width=crop_size, height=crop_size)
                ]
            )
        
##        self.unaligned = unaligned
##
        if mode == "CTtoMR":
            self.images = sorted(glob.glob(os.path.join(a_dir,'Images') + '/*.*'))
            self.labels = sorted(glob.glob(os.path.join(a_dir,'Label') + '/*.*'))
        elif mode == "MRtoCT":
            self.images = sorted(glob.glob(os.path.join(b_dir,'Images') + '/*.*'))
            self.labels = sorted(glob.glob(os.path.join(b_dir,'Label') + '/*.*'))
        

    def __getitem__(self, index):
##        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        image = nib.load(self.images[index % len(self.images)])
        name = self.images[index % len(self.images)]
        mask_load = nib.load(self.labels[index % len(self.images)])
        image = np.squeeze(image.get_fdata())
        image = ((image - image.min()) * 255. / (image.max() - image.min())).astype(np.uint8)
        
        mask_load = np.squeeze(mask_load.get_fdata())
        x,y = mask_load.shape
        mask = np.zeros((x,y))
        mask[mask_load == 500] = 1
        mask[mask_load == 205] = 2

        if self.clahe == True:
            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            image = Denoise(ApplyClaheColor(image))
        
        flipped = self.flipper(image=image,mask=mask)
        image = flipped['image']
        mask = flipped['mask']

        
        image_a = self.transform_im(Image.fromarray(image))
        mask_a = torch.from_numpy(mask)
        
        return {'A_img': image_a, 'A_mask': mask_a, 'file':name }

    def __len__(self):
        return len(self.images)

class SegGAN_MMWHS_UNet_Train(Dataset):
    def __init__(self, image_dir = None, transform_image=None,transform_mask=None, crop_size = 192, mode='MRtoCT', clahe = False):
        
        torch.manual_seed(42)
        random.seed(42)
        self.transform_im = transforms.Compose(transform_image)
        self.transform_label = transforms.Compose(transform_mask)
        self.clahe = clahe
        self.flipper = A.Compose(
            transforms = [
                A.CenterCrop(width=crop_size, height=crop_size)
                ]
            )
        
        if mode == "MRtoCT":
            mask_dir = r"CT_MR_2D_Dataset_DA-master\MR_withGT\Label"
            self.mask_dir = mask_dir
        elif mode == "CTtoMR":
            mask_dir = r"CT_MR_2D_Dataset_DA-master\CT_withGT\Label"
            self.mask_dir = mask_dir
            
        self.images = sorted(glob.glob(os.path.join(image_dir) + '/*.*'))
        self.labels = sorted(glob.glob(os.path.join(mask_dir) + '/*.*'))
        

    def __getitem__(self, index):
##        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        
        image = Image.open(self.images[index % len(self.images)])
        name = self.images[index % len(self.images)]
        head,tail = os.path.split(name)
        mask_name = tail.replace("img","lab")
        mask_name = mask_name.replace(".jpg",".nii.gz")
        mask_load = nib.load(os.path.join(self.mask_dir,mask_name))
        
        mask_load = np.squeeze(mask_load.get_fdata())
        x,y = mask_load.shape
        mask = np.zeros((x,y))
        mask[mask_load == 500] = 1
        mask[mask_load == 205] = 2

        if self.clahe == True:
            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            image = Denoise(ApplyClaheColor(image))
        
        flipped = self.flipper(image = mask)
        mask = flipped['image']

        
        image_a = self.transform_im(image)
        mask_a = torch.from_numpy(mask)
        
        return {'A_img': image_a, 'A_mask': mask_a, 'file':name }

    def __len__(self):
        return len(self.images)

class SegGAN_MMWHS_Test_CT_or_MR(Dataset):
    def __init__(self, transform_image=None,transform_mask=None, crop_size = 192, mode='MRtoCT', clahe = False):

        if mode == "MRtoCT":
            image_dir = r"CT_MR_2D_Dataset_DA-master\CT_withGT\Images"
            mask_dir = r"CT_MR_2D_Dataset_DA-master\CT_withGT\Label"
            
        elif mode == "CTtoMR":
            image_dir = r"CT_MR_2D_Dataset_DA-master\MR_withGT\Images"
            mask_dir = r"CT_MR_2D_Dataset_DA-master\MR_withGT\Label"
            
        torch.manual_seed(42)
        random.seed(42)
        self.transform_im = transforms.Compose(transform_image)
        self.transform_label = transforms.Compose(transform_mask)
        self.clahe = clahe
        self.flipper = A.Compose(
            transforms = [
                A.CenterCrop(width=crop_size, height=crop_size)
                ]
            )
        
##        self.unaligned = unaligned
##
        self.images = sorted(glob.glob(os.path.join(image_dir) + '/*.*'))
        self.labels = sorted(glob.glob(os.path.join(mask_dir) + '/*.*'))
        

    def __getitem__(self, index):
##        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        image = nib.load(self.images[index % len(self.images)])
        name = self.images[index % len(self.images)]
        mask_load = nib.load(self.labels[index % len(self.images)])
        image = np.squeeze(image.get_fdata())
        image = ((image - image.min()) * 255. / (image.max() - image.min())).astype(np.uint8)
        
        mask_load = np.squeeze(mask_load.get_fdata())
        x,y = mask_load.shape
        mask = np.zeros((x,y))
        mask[mask_load == 500] = 1
        mask[mask_load == 205] = 2

        if self.clahe == True:
            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            image = Denoise(ApplyClaheColor(image))
        
        flipped = self.flipper(image = image, mask=mask)
        image = flipped['image']
        mask = flipped['mask']

        
        image_a = self.transform_im(image)
        mask_a = torch.from_numpy(mask)
        
        return {'A_img': image_a, 'A_mask': mask_a, 'file':name }

    def __len__(self):
        return len(self.images)
