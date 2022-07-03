from torch.utils.data import Dataset
import os
import glob
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from imgaug import augmenters as iaa
import time
import imgaug as ia
import torch


class oasis_dataset(Dataset):
    def __init__(self, train, img_size, transforms=None, data_aug=0,
                 data_path='/home/Student/s4606685/summer_research/oasis-3'
                           '/png_data/T1w-png-converted/*.png'):
        self.img_size = img_size
        self.aug = data_aug
        self.img_paths = glob.glob(data_path)
        self.size = len(self.img_paths)
        self.transforms = transforms

    def __len__(self):
        return self.size

    def transform(self, img):
        # Function for data augmentation
        # 1) Affine Augmentations: Rotation (-15 to +15 degrees), Scaling, Flipping.
        # 2) Elastic deformations
        # 3) Intensity augmentations

        ia.seed(int(time.time()))  # Seed for random augmentations
        # Needed for iaa
        img = (img * 255).astype('uint8')

        if self.aug:  # Augmentation only performed on train set
            img = np.expand_dims(img, axis=0)

            seq_all = iaa.Sequential([
                iaa.Fliplr(0.5),  # Horizontal flips
                iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (0, 0), "y": (0, 0)},
                    rotate=(-10, 10),
                    shear=(0, 0)),  # Scaling, rotating
                iaa.ElasticTransformation(alpha=(0.0, 100.0), sigma=10.0),  # Elastic
                iaa.blur.AverageBlur(k=(0, 4)),  # Gausian blur
                iaa.LinearContrast((0.8, 1.2)),  # Contrast
                iaa.Multiply((0.8, 1.2), per_channel=1)  # Intensity
            ], random_order=True)

            images_aug = seq_all(images=img)  # Intensity and contrast only on input image

            img = np.squeeze(images_aug, axis=0)

        flip_tensor_trans = transforms.Compose([
            transforms.ToTensor()
        ])

        return flip_tensor_trans(img)

    def __getitem__(self, index):
        img_to_open = self.img_paths[index]
        img = Image.open(img_to_open)
        img = img.convert('L')
        img = img.resize((self.img_size, self.img_size))
        img = np.array(img)
        img = np.expand_dims(img, axis=-1)
        mask = torch.zeros(img.shape)
        mask[img > 0] = 1

        img_trans = self.transform(img)
        return img_trans, mask
