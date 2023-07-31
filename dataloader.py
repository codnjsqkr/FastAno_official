import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from random import shuffle
import random
import torchvision
import scipy.io as scio
import glob2

class VIDEO11(Dataset): 
    def __init__(self, data_list='./train.txt', num_of_img=3, num_of_skip=1, mode='train'):
        with open(os.path.join(data_list)) as f:
            self.mode = mode
            self.img_list = []
            self.num_of_img = num_of_img
            self.num_of_skip = num_of_skip
            self.transform_forlabel = transforms.Compose([transforms.ToTensor()])
            self.transform = transforms.Compose([transforms.ToTensor()])

            for line in f:
                line = line.replace("\n", "")
                lists = line.split(" ")
                root = lists[0]
                
                for i in range(1, len(lists)):
                    samples = []
                    for j in range(0, num_of_img):
                        try:
                            samples.append(lists[i + j*num_of_skip])
                        except:
                            pass
                    if len(samples) == num_of_img:
                        samples_str = ' '.join(samples)
                        self.img_list.append(root + " " + samples_str)
            
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_lists = (self.img_list[idx]).split(" ")
        aug_box = []
        real_box = []
        pred_box=[]
        draw_box = []
        img_sequence = list(range(self.num_of_img-1))
        shuffle(img_sequence)
        posW = random.randrange(0, 270)
        posH = random.randrange(30, 120)
        augtype = random.randrange(0,2) # 0: TMT, 1: SRT
        k = random.randrange(0, 4)

        for i in range(self.num_of_img-1): 
            img = cv2.imread(img_lists[0] + "/" + img_lists[i+1], 0)
            img = cv2.resize(img, dsize=(360, 240), interpolation=cv2.INTER_CUBIC)
            tensor_image = self.transform(img)
            tensor_for_gt = tensor_image.clone()
            tensor_image = torch.unsqueeze(tensor_image, dim=0)
            real_box.append(tensor_for_gt) #gt
            
            if self.mode == 'train':
                if (augtype == 0): # TMT
                    img2 = cv2.imread(img_lists[0] + "/" + img_lists[img_sequence[i]+1], 0) #for sequence augmentation
                    img2 = cv2.resize(img2, dsize=(360, 240), interpolation=cv2.INTER_CUBIC)
                    tensor_aug = self.transform(img2)
                    tensor_aug = torch.unsqueeze(tensor_aug, dim=0)
                    anopatch = tensor_aug[:,:,posH:(posH+90), posW:(posW+90)] #crop patch
                    tensor_image[:,:,posH:(posH+90), posW:(posW+90)] = anopatch #paste patch
                else:  #SRT
                    k = random.randrange(0, 4)
                    anopatch = tensor_image[:,:,posH:(posH+90), posW:(posW+90)]
                    anopatch = torch.rot90(anopatch, k, dims=(2,3))
                    tensor_image[:,:,posH:(posH+90), posW:(posW+90)] = anopatch
                aug_box.append(tensor_image) #augmented frame cuboid

        #================pred gt===========================
        pred_img = cv2.imread(img_lists[0] + "/" + img_lists[self.num_of_img], 0)
        pred_img = cv2.resize(pred_img, dsize=(360, 240), interpolation=cv2.INTER_CUBIC) # ped2
        pred_image = self.transform(pred_img)
        pred_torch = torch.unsqueeze(pred_image, dim=0)
        #==================================================

        if self.mode == 'train':
            result = torch.cat(aug_box, dim=0) #augmented
            result = result.transpose(0,1)
        real = torch.cat(real_box, dim=0) #gt            
        real = torch.unsqueeze(real, dim=0)
        pred_torch = pred_torch.transpose(0,1)

        folder_name = (img_lists[0].split("/"))[-1]
        if self.mode == 'train':
            return result, pred_torch, real, box
        else:
            return real, pred_torch, folder_name 

class Label_loader:
    def __init__(self, dataset, video_folders):
        self.data_root ='./data/'
        self.name = dataset
        self.mat_path = f'{self.data_root + self.name}/{self.name}.mat'
        self.video_folders = video_folders

    def __call__(self):
        if self.name == 'shanghaitech':
            gt = self.load_shanghaitech()
        else:
            gt = self.load_ucsd_avenue()
        return gt

    def load_ucsd_avenue(self):
        abnormal_events = scio.loadmat(self.mat_path, squeeze_me=True)['gt']
        all_gt = []
        for i in range(abnormal_events.shape[0]):
            length = len(os.listdir(self.video_folders[i]))
            sub_video_gt = np.zeros((length,), dtype=np.int8)
            one_abnormal = abnormal_events[i]
            if one_abnormal.ndim == 1:
                one_abnormal = one_abnormal.reshape((one_abnormal.shape[0], -1))
            for j in range(one_abnormal.shape[1]):
                start = one_abnormal[0, j] - 1
                end = one_abnormal[1, j]
                sub_video_gt[start: end] = 1
            all_gt.append(sub_video_gt)
        return all_gt

    def load_shanghaitech(self):
        np_list = glob2.glob(f'{self.data_root + self.name}/testing/test_frame_mask/*.npy')
        np_list.sort()
        gt = []
        for npy in np_list:
            gt.append(np.load(npy))
        return gt

def FrameCount(dataset):
    with open(f'{dataset}_test.txt', 'r') as file:
        file_contents = file.read()
    file_paths = file_contents.split()
    jpg_count = 0
    for file_path in file_paths:
        if file_path.endswith('.jpg'):
            jpg_count += 1
    return jpg_count