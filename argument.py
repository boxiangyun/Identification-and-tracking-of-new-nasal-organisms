import random

import cv2
from skimage.transform import AffineTransform, warp
import numpy as np
import skimage.io
import albumentations as A
import glob
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as AF

def resize(image, size=(128, 128)):
    return cv2.resize(image, size)


def add_gaussian_noise(x, sigma):
    x += np.random.randn(*x.shape) * sigma
    x = np.clip(x, 0., 1.)
    return x


def _evaluate_ratio(ratio):
    if ratio <= 0.:
        return False
    return np.random.uniform() < ratio

def apply_aug(aug, image, mask=None):
    # if mask is None:
    #     return aug(image=image)['image']
    # else:
    augment = aug(image=image,mask=mask)
    return augment['image'],augment['mask']

def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def adv_patch(image, patch, shape='circle'):# h and w must equal
    h, w = image.shape[0], image.shape[1]
    p_size = patch.shape[0]
    r = p_size//2
    x, y = np.clip(np.random.normal(h//2, 50, 2), r, h-r).astype(np.int)
    if shape=='patch':
        image[x-r:x+r, y-r:y+r] = patch
    elif shape=='circle':
        patch = create_circular_mask(p_size, p_size, radius=r)[:,:,None] * patch
        image = np.bitwise_not(create_circular_mask(h, w, center=(y, x), radius=r))[:,:,None] * image
        image[x-r :x+r, y-r:y+r] = image[x-r :x+r, y-r:y+r] + patch
    else:
        raise ValueError("Oops! That was no valid shape.Try again...")

    return image

class Transform:
    def __init__(self, train=True, target_size=None,
                 precess_aug_ration={"randomresizecrop": 0.,},
                 nature_aug_ration={"fog": 0., "rain": 0., "shadow":0., 'snow':.0, 'sun':0.},
                 general_aug_ration={"flip": 0., "blur": 0., "gauss":0., 'jitter':.0, 'grid':0., 'bright':0.},
                 ):
        self.train = train
        self.target_size = target_size
        self.precess_aug_ration=precess_aug_ration
        self.nature_aug_ration = nature_aug_ration
        self.general_aug_ration = general_aug_ration
        self.crop_transform = A.RandomResizedCrop(height=self.target_size, width=self.target_size,
                                             p=self.precess_aug_ration['randomresizecrop'])

        self.baseaug_transform = A.Compose([
            A.RandomFog(p=nature_aug_ration['fog']),
            A.RandomRain(p=nature_aug_ration['rain']),
            A.RandomShadow(p=nature_aug_ration['shadow']),
            A.RandomSunFlare(p=nature_aug_ration['sun'], src_radius=112),
            A.RandomSnow(p=nature_aug_ration['snow']),

            A.Flip(p=general_aug_ration['flip']),
            A.Blur(p=general_aug_ration['blur']),
            A.GaussNoise(p=general_aug_ration['gauss']),
            A.ColorJitter(p=general_aug_ration['jitter']),
            A.RandomBrightnessContrast(p=general_aug_ration['bright']),
        ])

    def __call__(self, example):
        if self.train:
            x, y = example
        else:
            x = example

        # albumentations...
        x = self.crop_transform(image=x)['image']
        x = self.baseaug_transform(image=x)['image']
        #resize
        if self.target_size is not None:
            if x.shape[0] != self.target_size or x.shape[1] != self.target_size:
                x = cv2.resize(x, (self.target_size, self.target_size))
                y = cv2.resize(y, (self.target_size, self.target_size))

        if self.train:
            return x, y
        else:
            return x
      
