import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import albumentations as A
import numpy as np
import cv2
import torch
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
    def __init__(self, a_dir, b_dir, transform_image=None,transform_mask=None, mode='train'):
        random.seed(42)
        torch.manual_seed(42)
        self.transform_im = transforms.Compose(transform_image)
        self.transform_label = transforms.Compose(transform_mask)
        self.flipper = A.Compose(
            transforms = [
                A.HorizontalFlip(p=0.5)]
            )
        
##        self.unaligned = unaligned
##        
        self.images_A = sorted(glob.glob(os.path.join(a_dir,'Images') + '/*.*'))
        self.labels_A = sorted(glob.glob(os.path.join(a_dir,'Label') + '/*.*'))
        self.images_B = sorted(glob.glob(os.path.join(b_dir,'Images') + '/*.*'))
        self.labels_B = sorted(glob.glob(os.path.join(b_dir,'Label') + '/*.*'))
        

    def __getitem__(self, index):
##        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        image_a = np.asarray(Image.open(self.images_A[index % len(self.images_A)]))
        mask_a = np.asarray(Image.open(self.labels_A[index % len(self.labels_A)]))
        
        flipped = self.flipper(image=image_a,mask=mask_a)
        image_a = flipped['image']
        mask_a = flipped['mask']

        rand = random.randint(0, len(self.images_B) - 1)
        image_b = np.asarray(Image.open(self.images_B[rand]))
        mask_b = np.asarray(Image.open(self.labels_B[rand]))
        flipped = self.flipper(image=image_b,mask=mask_b)
        
        image_b = flipped['image']
        mask_b = flipped['mask']
        image_a = self.transform_im(Image.fromarray(image_a))
        mask_a = self.transform_label(Image.fromarray(mask_a))

        image_b = self.transform_im(Image.fromarray(image_b))
        mask_b = self.transform_label(Image.fromarray(mask_b))
        
        
##            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
##        else:
##            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A_img': image_a, 'A_mask': mask_a, 'B_img': image_b, 'B_mask': mask_b}

    def __len__(self):
        return len(self.images_A)

        
##            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
##        else:
##            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A_img': image_a, 'A_mask': mask_a, 'B_img': image_b, 'B_mask': mask_b}

    def __len__(self):
        return max(len(self.images_A), len(self.images_B))
    

class SegGAN_Dataset_v2(Dataset):
    def __init__(self, a_dir, b_dir, transform_image=None,transform_mask=None, resize_size = None, crop_size = None, mode='train', clahe = True):
        random.seed(42)
        torch.manual_seed(42)

        self.transform_im = transforms.Compose(transform_image)
        self.transform_label = transforms.Compose(transform_mask)
        self.clahe = clahe
        self.flipper = A.Compose(
            transforms = [
                A.Resize(width = resize_size, height= resize_size),
                A.RandomCrop(width=crop_size, height=crop_size)
                ]
            )
        
##        self.unaligned = unaligned
##        
        self.images_A = sorted(glob.glob(os.path.join(a_dir,'Images') + '/*.*'))
        self.labels_A = sorted(glob.glob(os.path.join(a_dir,'Label') + '/*.*'))
        self.images_B = sorted(glob.glob(os.path.join(b_dir,'Images') + '/*.*'))
        self.labels_B = sorted(glob.glob(os.path.join(b_dir,'Label') + '/*.*'))
        

    def __getitem__(self, index):
##        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        image_a = np.asarray(Image.open(self.images_A[index % len(self.images_A)]))
        mask_a = np.asarray(Image.open(self.labels_A[index % len(self.labels_A)]))
        filename = self.images_A[index % len(self.images_A)]
        if self.clahe == True:
            image_a = cv2.cvtColor(image_a,cv2.COLOR_RGB2BGR)
            image_a = Denoise(ApplyClaheColor(image_a))
        
        flipped = self.flipper(image=image_a,mask=mask_a)
        image_a = flipped['image']
        mask_a = flipped['mask']

        rand = random.randint(0, len(self.images_B) - 1)
        image_b = np.asarray(Image.open(self.images_B[rand]))
        mask_b = np.asarray(Image.open(self.labels_B[rand]))

        if self.clahe == True:        
            image_b = cv2.cvtColor(image_b,cv2.COLOR_RGB2BGR)
            image_b = Denoise(ApplyClaheColor(image_b))

        flipped = self.flipper(image=image_b,mask=mask_b)
        
        image_b = flipped['image']
        mask_b = flipped['mask']
        
        image_a = self.transform_im(Image.fromarray(image_a))
        mask_a = self.transform_label(Image.fromarray(mask_a))

        image_b = self.transform_im(Image.fromarray(image_b))
        mask_b = self.transform_label(Image.fromarray(mask_b))
        
        return {'A_img': image_a, 'A_mask': mask_a, 'B_img': image_b, 'B_mask': mask_b}

    def __len__(self):
        return max(len(self.images_A), len(self.images_B))

    
class SegGAN_Dataset_v2_noBMask(Dataset):
    def __init__(self, a_dir, b_dir, transform_image=None,transform_mask=None, resize_size = None, crop_size = None, mode='train', clahe = True):
        random.seed(42)
        torch.manual_seed(42)
        self.transform_im = transforms.Compose(transform_image)
        self.transform_label = transforms.Compose(transform_mask)
        self.clahe = clahe
        self.flipper = A.Compose(
            transforms = [
                A.Resize(width = resize_size, height= resize_size),
                A.RandomCrop(width=crop_size, height=crop_size)
                ]
            )
        
##        self.unaligned = unaligned
##        
        self.images_A = sorted(glob.glob(os.path.join(a_dir,'Images') + '/*.*'))
        self.labels_A = sorted(glob.glob(os.path.join(a_dir,'Label') + '/*.*'))
        self.images_B = sorted(glob.glob(os.path.join(b_dir,'Images') + '/*.*'))
        

    def __getitem__(self, index):
##        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        image_a = np.asarray(Image.open(self.images_A[index % len(self.images_A)]))
        mask_a = np.asarray(Image.open(self.labels_A[index % len(self.labels_A)]))
        file_name = self.images_A[index % len(self.images_A)]
        
        if self.clahe == True:
            image_a = cv2.cvtColor(image_a,cv2.COLOR_RGB2BGR)
            image_a = Denoise(ApplyClaheColor(image_a))
        
        flipped = self.flipper(image=image_a,mask=mask_a)
        image_a = flipped['image']
        mask_a = flipped['mask']

        rand = random.randint(0, len(self.images_B) - 1)
        image_b = np.asarray(Image.open(self.images_B[rand]))

        if self.clahe == True:        
            image_b = cv2.cvtColor(image_b,cv2.COLOR_RGB2BGR)
            image_b = Denoise(ApplyClaheColor(image_b))

        flipped = self.flipper(image=image_b)
        
        image_b = flipped['image']
        
        image_a = self.transform_im(Image.fromarray(image_a))
        mask_a = self.transform_label(Image.fromarray(mask_a))

        image_b = self.transform_im(Image.fromarray(image_b))
        
        return {'A_img': image_a, 'A_mask': mask_a, 'B_img': image_b}

    def __len__(self):
        return len(self.images_A)

class SegGAN_Dataset_v2_noCLAHE(Dataset):
    def __init__(self, a_dir, b_dir, transform_image=None,transform_mask=None, mode='train'):
        self.transform_im = transforms.Compose(transform_image)
        self.transform_label = transforms.Compose(transform_mask)
        self.flipper = A.Compose(
            transforms = [
                A.Resize(height=512,width=512),
                A.RandomCrop(height=256, width=256),
                A.VerticalFlip(p=0.5)]
            )
        
##        self.unaligned = unaligned
##        
        self.images_A = sorted(glob.glob(os.path.join(a_dir,'Images') + '/*.*'))
        self.labels_A = sorted(glob.glob(os.path.join(a_dir,'Label') + '/*.*'))
        self.images_B = sorted(glob.glob(os.path.join(b_dir,'Images') + '/*.*'))
        self.labels_B = sorted(glob.glob(os.path.join(b_dir,'Label') + '/*.*'))
        

    def __getitem__(self, index):
##        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        image_a = np.asarray(Image.open(self.images_A[index % len(self.images_A)]))
        mask_a = np.asarray(Image.open(self.labels_A[index % len(self.labels_A)]))

        
        flipped = self.flipper(image=image_a,mask=mask_a)
        image_a = flipped['image']
        mask_a = flipped['mask']

        rand = random.randint(0, len(self.images_B) - 1)
        image_b = np.asarray(Image.open(self.images_B[rand]))
        mask_b = np.asarray(Image.open(self.labels_B[rand]))
        
        flipped = self.flipper(image=image_b,mask=mask_b)
        
        image_b = flipped['image']
        mask_b = flipped['mask']
        
        image_a = self.transform_im(Image.fromarray(image_a))
        mask_a = self.transform_label(Image.fromarray(mask_a))

        image_b = self.transform_im(Image.fromarray(image_b))
        mask_b = self.transform_label(Image.fromarray(mask_b))
        
        return {'A_img': image_a, 'A_mask': mask_a, 'B_img': image_b, 'B_mask': mask_b}

    def __len__(self):
        return max(len(self.images_A), len(self.images_B))

class SegGAN_Dataset_Test_A2B(Dataset):
    def __init__(self, a_dir, transform_image=None, mode='test'):
        self.transform_im = transforms.Compose(transform_image)
        self.flipper = A.Compose(
            transforms = [
                A.HorizontalFlip(p=0.5)]
            )
        
##        self.unaligned = unaligned
##        
        self.images_A = sorted(glob.glob(os.path.join(a_dir,'Images') + '/*.*'))
        

    def __getitem__(self, index):
##        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        
        file_name = self.images_A[index % len(self.images_A)]
        image_a = np.asarray(Image.open(self.images_A[index % len(self.images_A)]))        
        flipped = self.flipper(image=image_a)
        image_a = flipped['image']
        image_a = self.transform_im(Image.fromarray(image_a))

        return {'A_img': image_a, 'file': file_name}

    def __len__(self):
        return len(self.images_A)
    
class SegGAN_Dataset_Test_A2B_No_Flip(Dataset):
    def __init__(self, a_dir, transform_image=None, mode='test'):
        self.transform_im = transforms.Compose(transform_image)
        self.images_A = sorted(glob.glob(os.path.join(a_dir,'Images') + '/*.*'))
        

    def __getitem__(self, index):
##        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        
        file_name = self.images_A[index % len(self.images_A)]
        image_a = np.asarray(Image.open(self.images_A[index % len(self.images_A)]))
        image_a = cv2.cvtColor(image_a,cv2.COLOR_RGB2BGR)
        image_a = Denoise(ApplyClaheColor(image_a))
        image_a = self.transform_im(Image.fromarray(image_a))

        return {'A_img': image_a, 'file': file_name}

    def __len__(self):
        return len(self.images_A)

class SegGAN_DRIVE_Patches_To_STARE_With_Label_noBMask(Dataset):
    def __init__(self, a_dir, b_dir, transform_image=None, transform_label=None, resize_scale = 512, mode='test'):
        torch.manual_seed(42)
        random.seed(42)
        self.transform_im = transforms.Compose(transform_image)
        self.transform_label = transforms.Compose(transform_label)
        
        self.images_A = sorted(glob.glob(os.path.join(a_dir,'Images') + '/*.*'))
        self.labels_A = sorted(glob.glob(os.path.join(a_dir,'Label') + '/*.*'))
        self.images_B = sorted(glob.glob(os.path.join(b_dir,'Images') + '/*.*'))
        self.flipper = A.Compose(
            transforms = [
            A.Resize(height=resize_scale,width=resize_scale)]
        )
        

    def __getitem__(self, index):
##        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        
        file_name = self.images_A[index % len(self.images_A)]
        image_a = np.asarray(Image.open(self.images_A[index % len(self.images_A)]))
        mask_a = np.asarray(Image.open(self.labels_A[index % len(self.labels_A)]))
        flipped = self.flipper(image=image_a,mask=mask_a)
        image_a = flipped['image']
        mask_a = flipped['mask']
        image_a = self.transform_im(Image.fromarray(image_a))
        mask_a = self.transform_label(Image.fromarray(mask_a))

        rand = random.randint(0, len(self.images_B) - 1)
        image_b = np.asarray(Image.open(self.images_B[rand]))

        flipped = self.flipper(image=image_b)
        
        image_b = flipped['image']
        image_b = self.transform_im(Image.fromarray(image_b))

        return {'A_img': image_a, 'A_mask':mask_a, 'B_img': image_b, 'file': file_name}

    def __len__(self):
        return len(self.images_A)

class SegGAN_STARE_Patches_To_DRIVE_With_Label(Dataset):
    def __init__(self, a_dir, b_dir, transform_image=None, transform_label=None, resize_scale = 512, mode='test'):
        torch.manual_seed(42)
        random.seed(42)
        self.transform_im = transforms.Compose(transform_image)
        self.transform_label = transforms.Compose(transform_label)
        
        self.images_A = sorted(glob.glob(os.path.join(a_dir,'Images') + '/*.*'))
        self.labels_A = sorted(glob.glob(os.path.join(a_dir,'Label') + '/*.*'))
        self.images_B = sorted(glob.glob(os.path.join(b_dir,'Images') + '/*.*'))
        self.labels_B = sorted(glob.glob(os.path.join(b_dir,'Label') + '/*.*'))

        self.flipper = A.Compose(
            transforms = [
            A.Resize(height=resize_scale,width=resize_scale)]
        )
        

    def __getitem__(self, index):
##        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        
        file_name = self.images_A[index % len(self.images_A)]
        image_a = np.asarray(Image.open(self.images_A[index % len(self.images_A)]))
        mask_a = np.asarray(Image.open(self.labels_A[index % len(self.labels_A)]))
        flipped = self.flipper(image=image_a,mask=mask_a)
        image_a = flipped['image']
        mask_a = flipped['mask']
        image_a = self.transform_im(Image.fromarray(image_a))
        mask_a = self.transform_label(Image.fromarray(mask_a))

        rand = random.randint(0, len(self.images_B) - 1)

        image_b = np.asarray(Image.open(self.images_B[rand]))
        mask_b = np.asarray(Image.open(self.labels_B[rand]))

        flipped = self.flipper(image=image_b,mask=mask_b)
        
        image_b = flipped['image']
        mask_b = flipped['mask']
        
        image_b = self.transform_im(Image.fromarray(image_b))
        mask_b = self.transform_label(Image.fromarray(mask_b))

        return {'A_img': image_a, 'A_mask':mask_a, 'B_img': image_b, 'B_mask': mask_b, 'file': file_name}

    def __len__(self):
        return len(self.images_A)

class SegGAN_DRIVE_Patches_To_STARE_With_Label(Dataset):
    def __init__(self, a_dir=r"C:\Users\Paolo\Documents\Eye_Seg\DRIVE\training", transform_image=None, transform_label=None, resize_scale = 512, mode='test'):
        self.transform_im = transforms.Compose(transform_image)
        self.transform_label = transforms.Compose(transform_label)
        self.images_A = sorted(glob.glob(os.path.join(a_dir,'Images') + '/*.*'))
        self.labels_A = sorted(glob.glob(os.path.join(a_dir,'Label') + '/*.*'))
        self.flipper = A.Compose(
            transforms = [
            A.Resize(height=resize_scale,width=resize_scale)]
        )
        

    def __getitem__(self, index):
##        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        
        file_name = self.images_A[index % len(self.images_A)]
        image_a = np.asarray(Image.open(self.images_A[index % len(self.images_A)]))
        mask_a = np.asarray(Image.open(self.labels_A[index % len(self.labels_A)]))
        flipped = self.flipper(image=image_a,mask=mask_a)
        image_a = flipped['image']
        mask_a = flipped['mask']
        image_a = self.transform_im(Image.fromarray(image_a))
        mask_a = self.transform_label(Image.fromarray(mask_a))
        
        return {'A_img': image_a, 'A_mask':mask_a, 'file': file_name}

    def __len__(self):
        return len(self.images_A)

class SegGAN_STARE_Patches_To_DRIVE_Without_Label(Dataset):
    def __init__(self, a_dir, transform_image=None, resize_scale = 512, mode='test'):
        self.transform_im = transforms.Compose(transform_image)
        self.images_A = sorted(glob.glob(os.path.join(a_dir,'Images') + '/*.*'))
        self.flipper = A.Compose(
            transforms = [
            A.Resize(height=resize_scale,width=resize_scale)]
        )
        

    def __getitem__(self, index):
##        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        
        file_name = self.images_A[index % len(self.images_A)]
        image_a = np.asarray(Image.open(self.images_A[index % len(self.images_A)]))
        flipped = self.flipper(image=image_a)
        image_a = flipped['image']
        image_a = self.transform_im(Image.fromarray(image_a))

        return {'A_img': image_a, 'file': file_name}

    def __len__(self):
        return len(self.images_A)

class SegGAN_Dataset_Test_A2B_No_Flip_noCLAHE(Dataset):
    def __init__(self, a_dir, transform_image=None, resize_scale = 512, mode='test'):
        self.transform_im = transforms.Compose(transform_image)
        self.images_A = sorted(glob.glob(os.path.join(a_dir,'Images') + '/*.*'))
        self.flipper = A.Compose(
            transforms = [
            A.Resize(height=resize_scale,width=resize_scale)]
        )
        

    def __getitem__(self, index):
##        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        
        file_name = self.images_A[index % len(self.images_A)]
        image_a = np.asarray(Image.open(self.images_A[index % len(self.images_A)]))
        flipped = self.flipper(image=image_a)
        image_a = flipped['image']
        image_a = self.transform_im(Image.fromarray(image_a))

        return {'A_img': image_a, 'file': file_name}

    def __len__(self):
        return len(self.images_A)
    
class SegGAN_Dataset_STARE_Labelled_To_DRIVE(Dataset):
    def __init__(self, a_dir, transform_image=None,transform_masks = None, mode='test'):
        self.transform_im = transforms.Compose(transform_image)
        self.transform_mask = transforms.Compose(transform_masks)
        self.images_A = sorted(glob.glob(os.path.join(a_dir,'Images') + '/*.*'))
        self.labels_A = sorted(glob.glob(os.path.join(a_dir,'Label') + '/*.*'))

    def __getitem__(self, index):
##        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        
        file_name = self.images_A[index % len(self.images_A)]
        image_a = np.asarray(Image.open(self.images_A[index % len(self.images_A)]))
        label_a = np.asarray(Image.open(self.labels_A[index % len(self.images_A)]))
        image_a = self.transform_im(Image.fromarray(image_a))
        label_a = self.transform_mask(Image.fromarray(label_a))

        return {'A_img': image_a, 'A_mask': label_a, 'file': file_name}

    def __len__(self):
        return len(self.images_A)
class SegGAN_Dataset_STARE_Labelled_To_DRIVE_CLAHE(Dataset):
    def __init__(self, a_dir, transform_image=None,transform_masks = None, mode='test'):
        self.transform_im = transforms.Compose(transform_image)
        self.transform_mask = transforms.Compose(transform_masks)
        self.images_A = sorted(glob.glob(os.path.join(a_dir,'Images') + '/*.*'))
        self.labels_A = sorted(glob.glob(os.path.join(a_dir,'Label') + '/*.*'))

    def __getitem__(self, index):
##        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        
        file_name = self.images_A[index % len(self.images_A)]
        image_a = np.asarray(Image.open(self.images_A[index % len(self.images_A)]))
        image_a = cv2.cvtColor(image_a,cv2.COLOR_RGB2BGR)
        image_a = Denoise(ApplyClaheColor(image_a))

        label_a = np.asarray(Image.open(self.labels_A[index % len(self.images_A)]))
        image_a = self.transform_im(Image.fromarray(image_a))
        label_a = self.transform_mask(Image.fromarray(label_a))

        return {'A_img': image_a, 'A_mask': label_a, 'file': file_name}

    def __len__(self):
        return len(self.images_A)
