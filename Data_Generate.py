#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch.utils.data.dataset import Dataset
import glob
import numpy as np
import cv2
import os
import warnings
warnings.filterwarnings('ignore')

def mask2label(mask):#b, 4, h, w
    shape = mask.shape
    mask = mask.reshape(shape[0], shape[1], -1)
    mask_count = np.count_nonzero(mask, axis=-1)
    mask = mask_count.argmax(1)

    return (np.eye(4)[mask]).astype(int)

class Data_Generate_Base(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None, cut=1024, target_size=576):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.class_m = {'NP':0, 'NIP':1, 'FS':2, 'TM':3}
        self.cut = cut
        self.target_size = target_size
#         self.target_size = target_size

    def __getitem__(self, index):
        img = cv2.imread(self.img_paths[index])[:,:,::-1]
        mask = cv2.imread(self.mask_paths[index], 0)
        shape = img.shape

        if shape[1] >= self.cut:
            img = img[shape[0]//2-self.cut//2: shape[0]//2+self.cut//2, shape[1]//2-self.cut//2: shape[1]//2+self.cut//2]
            mask = mask[shape[0]//2-self.cut//2: shape[0]//2+self.cut//2, shape[1]//2-self.cut//2: shape[1]//2+self.cut//2]
            img = cv2.resize(img, (self.target_size, self.target_size))
        else:
            img = img[shape[0]//2-self.target_size//2: shape[0]//2+self.target_size//2, shape[1]//2-self.target_size//2: shape[1]//2+self.target_size//2]
            mask = mask[shape[0]//2-self.target_size//2: shape[0]//2+self.target_size//2, shape[1]//2-self.target_size//2: shape[1]//2+self.target_size//2]

        if self.transform is not None:
            img, mask = self.transform((img, mask))

        shape = img.shape
        class_index = self.class_m[self.mask_paths[index].split('/')[-3]]
        mask_onehot = np.zeros((shape[0], shape[1], 4))
        mask_onehot[:, :, class_index] = mask
        if class_index == 1:
            mask_onehot[:, :, 3] = mask
        img = (np.transpose(img, (2, 0, 1))/255).astype(np.float32)# c h w
        mask_onehot = (np.transpose(mask_onehot, (2, 0, 1))/255).astype(np.float32)# c h w
        # print(f"we get {mask2label(mask_onehot[None])}, {class_index}")
        return img, mask_onehot

    def __len__(self):
        return len(self.img_paths)

class Data_Generate_HIP(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None, cut=1024, target_size=576):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        # self.class_m = {'NP':0, 'NIP':1, 'FS':2, 'TM':3}
        self.cut = cut
        self.target_size = target_size
#         self.target_size = target_size

    def __getitem__(self, index):
        img = cv2.imread(self.img_paths[index])[:,:,::-1]
        mask = cv2.imread(self.mask_paths[index], 0)
        shape = img.shape

        if shape[1] >= self.cut:
            img = img[shape[0]//2-self.cut//2: shape[0]//2+self.cut//2, shape[1]//2-self.cut//2: shape[1]//2+self.cut//2]
            mask = mask[shape[0]//2-self.cut//2: shape[0]//2+self.cut//2, shape[1]//2-self.cut//2: shape[1]//2+self.cut//2]
            img = cv2.resize(img, (self.target_size, self.target_size))
        else:
            img = img[shape[0]//2-self.target_size//2: shape[0]//2+self.target_size//2, shape[1]//2-self.target_size//2: shape[1]//2+self.target_size//2]
            mask = mask[shape[0]//2-self.target_size//2: shape[0]//2+self.target_size//2, shape[1]//2-self.target_size//2: shape[1]//2+self.target_size//2]

        if self.transform is not None:
            img, mask = self.transform((img, mask))
        mask = mask[:,:,None]
        shape = img.shape
        # class_index = self.class_m[self.mask_paths[index].split('/')[-3]]
        # mask_onehot = np.zeros((shape[0], shape[1], 4))
        # mask_onehot[:, :, class_index] = mask
        # if class_index == 1:
        #     mask_onehot[:, :, 3] = mask
        img = (np.transpose(img, (2, 0, 1))/255).astype(np.float32)# c h w
        mask = (np.transpose(mask, (2, 0, 1))/255).astype(np.float32)# c h w
        # print(f"we get {mask2label(mask_onehot[None])}, {class_index}")
        return img, mask

    def __len__(self):
        return len(self.img_paths)

#class Data_Generate_Inference(Dataset):
    # def __init__(self, img_paths, transform=None, target_size=224):
    #     self.img_paths = sorted(glob.glob(img_paths + '/*.png'), key=lambda x: int(x.split('/')[-1].split('.')[0]))
    #     print(len(self.img_paths))
    #     self.transform = transform
    #     self.target_size = target_size
    #
    # def __getitem__(self, index):
    #     img_path = self.img_paths[index]
    #     img = cv2.imread(img_path)[:, :, ::-1]
    #     if self.transform is not None:
    #         img = self.transform(img)
    #
    #     if img.shape[0] != self.target_size or img.shape[1] != self.target_size:
    #         img = cv2.resize(img, (self.target_size, self.target_size))
    #     img = (np.transpose(img, (2, 0, 1))/255).astype(np.float32)# c h w
    #     return img
    #
    # def __len__(self):
    #     return len(self.img_paths)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os
    import json
    import cv2
    from argument import Transform

    with open('/home/ubuntu/T/Nose/all_data/image/zyn.json', 'r') as f:
        file_names = json.load(f)

    root_path = "/home/ubuntu/T/Nose/all_data/image"
    train_img_paths, val_img_paths, test_img_paths = [], [], []
    train_mask_paths, val_mask_paths, test_mask_paths = [], [], []
    for c in ['NP', 'NIP', 'FS', 'TM']:
        train_img_paths += [os.path.join(root_path, c, 'images', f"{i}.jpg") for i in file_names[c]['train']]
        train_mask_paths += [os.path.join(root_path, c, 'masks', f"{i}.png") for i in file_names[c]['train']]
        val_img_paths += [os.path.join(root_path, c, 'images', f"{i}.jpg") for i in file_names[c]['val']]
        val_mask_paths += [os.path.join(root_path, c, 'masks', f"{i}.png") for i in file_names[c]['val']]
        test_img_paths += [os.path.join(root_path, c, 'images', f"{i}.jpg") for i in file_names[c]['test']]
        test_mask_paths += [os.path.join(root_path, c, 'masks', f"{i}.png") for i in file_names[c]['test']]

    np.random.seed(42)
    np.random.shuffle(train_img_paths)
    np.random.seed(42)
    np.random.shuffle(train_mask_paths)
    np.random.seed(42)
    np.random.shuffle(val_img_paths)
    np.random.seed(42)
    np.random.shuffle(val_mask_paths)
    np.random.seed(42)
    np.random.shuffle(test_img_paths)
    np.random.seed(42)
    np.random.shuffle(test_mask_paths)

    root_path = '/home/ubuntu/T/kaggle/Domain_workshop/train_phase1/train/train'
    transform = Transform(target_size=576, nature_aug_ration = {"fog": 0., "rain": 0., "shadow":0., 'snow':0., 'sun':0.},
                                    general_aug_ration = {"flip": 0.2, "blur": 0.1, "gauss":0.1, 'jitter':0., 'bright':0.2})

    train_db = Data_Generate_Base(train_img_paths, train_mask_paths, transform=transform, cut=1024)
    imgs, labels = train_db[0]
    print(imgs.shape, labels.shape, imgs.max(), imgs.min(), labels.max(), labels.min())

    f, ax = plt.subplots(4, 5, figsize=(25, 25))
    for i in range(4):
        img, labels = train_db[i]
        img = np.transpose(img, (1, 2, 0))
        ax[i, 0].imshow(img)
        ax[i, 0].set_title(f"orign image {train_db.img_paths[i].split('/')[-1][:-5]}")
        for j in range(0, 5):
            img, label = train_db[i]
            img = np.transpose(img, (1, 2, 0))
            # aug_img = transform(image=img)['image']
            ax[i, j].imshow(img)
            ax[i, j].set_title(f'aug:{j}')
    plt.show()
