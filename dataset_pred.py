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

class VIDEO(Dataset): #rotation augmentation, degree all different for n frames
    def __init__(self, data_list='./train.txt', num_of_img=3, num_of_skip=1, mode='train'):
        with open(os.path.join(data_list)) as f:
            self.mode = mode
            self.img_list = []
            self.num_of_img = num_of_img
            self.num_of_skip = num_of_skip
            self.transform_forlabel = transforms.Compose([transforms.ToTensor()])
            self.transform = transforms.Compose([transforms.ToTensor()])
            # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

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
        img_box = []
        real_box = []
        pred_box=[]
        # print(self.img_list)

        # rotation = np.zeros(self.num_of_img) #recon
        rotation = np.zeros(self.num_of_img-1)

        posW = random.randrange(0, 300)
        posH = random.randrange(60, 120)
        # posW = random.randrange(0, 198)
        # posH = random.randrange(0, 132)
        # posW = random.randrange(0, 560)
        # posH = random.randrange(70, 210)

        for i in range(self.num_of_img-1): #pred의 경우 -1, recon의 경우 삭제해야함
            img = cv2.imread(img_lists[0] + "/" + img_lists[i+1], 0)
            img = cv2.resize(img, dsize=(360, 240), interpolation=cv2.INTER_CUBIC) # ped2
            # img = cv2.resize(img, dsize=(220, 154), interpolation=cv2.INTER_CUBIC) # ped1
            # img = cv2.resize(img, dsize=(630, 350), interpolation=cv2.INTER_CUBIC) # avn
            tensor_image = self.transform(img)
            tensor2 = tensor_image.clone()
            real_box.append(tensor2) #gt

            k = random.randrange(0, 4)
            rotation[i]=k

            # print(tensor_image.shape)
            anopatch = tensor_image[:,posH:(posH+60), posW:(posW+60)] #ped2: 30, ped1:22, avenue:70
            anopatch = torch.rot90(anopatch, k, dims=(1,2))
            tensor_image[:,posH:(posH+60), posW:(posW+60)] = anopatch

            img_box.append(tensor_image) #augmented


        #================pred gt===========================
        pred_img = cv2.imread(img_lists[0] + "/" + img_lists[self.num_of_img], 0)
        pred_img = cv2.resize(pred_img, dsize=(360, 240), interpolation=cv2.INTER_CUBIC) # ped2
        pred_image = self.transform(pred_img)
        pred_torch = torch.unsqueeze(pred_image, dim=1)
        #==================================================


        np_label = np.zeros((4, self.num_of_img-1)) #recon, pred
        
        for x in range(self.num_of_img-1): #recon, pred
            y = int(rotation[x])
            np_label[y][x] = 1
        
        label = self.transform_forlabel(np_label)
        label = torch.squeeze(label, 0)

        result = torch.cat(img_box, dim=0) #augmented
        real = torch.cat(real_box, dim=0) #gt

        result = torch.unsqueeze(result, dim=1)
        real = torch.unsqueeze(real, dim=1)

        folder_name = (img_lists[0].split("/"))[-1]
        if self.mode == 'train':
            # return result, real, label.long() #recon
            return result, pred_torch, real, label.long()
        else:
            # return real, real, folder_name #기본 rotation만 준 코드
            return real, pred_torch, folder_name #task5 (sequence shuffle 코드)

class VIDEO2(Dataset): #sequence shuffle
    def __init__(self, data_list='./train.txt', num_of_img=3, num_of_skip=1, mode='train'):
        with open(os.path.join(data_list)) as f:
            self.mode = mode
            self.img_list = []
            self.num_of_img = num_of_img
            self.num_of_skip = num_of_skip
            self.transform_forlabel = transforms.Compose([transforms.ToTensor()])
            self.transform = transforms.Compose([transforms.ToTensor()])
            # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

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
        # print(self.img_list)

        # rotation = np.zeros(self.num_of_img) #recon
        img_sequence = list(range(self.num_of_img))
        shuffle(img_sequence)

        posW = random.randrange(0, 300)
        posH = random.randrange(60, 120)
        # posW = random.randrange(0, 198)
        # posH = random.randrange(0, 132)
        # posW = random.randrange(0, 560)
        # posH = random.randrange(70, 210)

        for i in range(self.num_of_img-1): #pred의 경우 -1, recon의 경우 삭제해야함
            img = cv2.imread(img_lists[0] + "/" + img_lists[i+1], 0)
            img = cv2.resize(img, dsize=(360, 240), interpolation=cv2.INTER_CUBIC) # ped2
            # img = cv2.resize(img, dsize=(220, 154), interpolation=cv2.INTER_CUBIC) # ped1
            # img = cv2.resize(img, dsize=(630, 350), interpolation=cv2.INTER_CUBIC) # avn
            tensor_image = self.transform(img)
            tensor_for_gt = tensor_image.clone()
            real_box.append(tensor_for_gt) #gt

            img2 = cv2.imread(img_lists[0] + "/" + img_lists[img_sequence[i]+1], 0) #for sequence augmentation
            img2 = cv2.resize(img2, dsize=(360, 240), interpolation=cv2.INTER_CUBIC)
            tensor_aug = self.transform(img2)
            anopatch = tensor_aug[:,posH:(posH+60), posW:(posW+60)] #crop patch
            tensor_image[:,posH:(posH+60), posW:(posW+60)] = anopatch #paste patch


            aug_box.append(tensor_image) #augmented frame cuboid

        #================pred gt===========================
        pred_img = cv2.imread(img_lists[0] + "/" + img_lists[self.num_of_img], 0)
        pred_img = cv2.resize(pred_img, dsize=(360, 240), interpolation=cv2.INTER_CUBIC) # ped2
        pred_image = self.transform(pred_img)
        pred_torch = torch.unsqueeze(pred_image, dim=1)
        #==================================================


        result = torch.cat(aug_box, dim=0) #augmented
        real = torch.cat(real_box, dim=0) #gt

        result = torch.unsqueeze(result, dim=1)
        real = torch.unsqueeze(real, dim=1)

        folder_name = (img_lists[0].split("/"))[-1]
        if self.mode == 'train':
            # return result, real, label.long() #recon
            return result, pred_torch, real
        else:
            # return real, real, folder_name #기본 rotation만 준 코드
            return real, pred_torch, folder_name #task5 (sequence shuffle 코드)



'''====================================='''
'''  3. sequence or rotation '''
'''====================================='''
class VIDEO3(Dataset): #sequence shuffle
    def __init__(self, data_list='./train.txt', num_of_img=3, num_of_skip=1, mode='train'):
        with open(os.path.join(data_list)) as f:
            self.mode = mode
            self.img_list = []
            self.num_of_img = num_of_img
            self.num_of_skip = num_of_skip
            self.transform_forlabel = transforms.Compose([transforms.ToTensor()])
            self.transform = transforms.Compose([transforms.ToTensor()])
            # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

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
        # print(self.img_list)

        img_sequence = list(range(self.num_of_img))
        shuffle(img_sequence)

        # posW = random.randrange(0, 300)
        # posH = random.randrange(60, 120)
        posW = random.randrange(0, 270)
        posH = random.randrange(30, 150)
        # posW = random.randrange(0, 198)
        # posH = random.randrange(0, 132)
        # posW = random.randrange(0, 560)
        # posH = random.randrange(70, 210)
        augtype = random.randrange(0,2) #0이면 셔플, 1이면 로테이션

        for i in range(self.num_of_img-1): #pred의 경우 -1, recon의 경우 삭제해야함
            img = cv2.imread(img_lists[0] + "/" + img_lists[i+1], 0)
            img = cv2.resize(img, dsize=(360, 240), interpolation=cv2.INTER_CUBIC) # ped2
            # img = cv2.resize(img, dsize=(220, 154), interpolation=cv2.INTER_CUBIC) # ped1
            # img = cv2.resize(img, dsize=(630, 350), interpolation=cv2.INTER_CUBIC) # avn
            tensor_image = self.transform(img)
            tensor_for_gt = tensor_image.clone()
            real_box.append(tensor_for_gt) #gt

            img_drawbox = img
            # print(img_drawbox.shape)
            img_drawbox[posH:(posH+60), posW:(posW+1)] = 255 #왼쪽 column
            img_drawbox[posH:(posH+60), (posW+60):(posW+61)] = 255 #오른쪽 column
            img_drawbox[posH:(posH+1), posW:(posW+60)] = 255 #위쪽 row
            img_drawbox[(posH+60):(posH+61), posW:(posW+61)] = 255 #아래쪽 row
            tensor_drawbox = self.transform(img_drawbox)
            draw_box.append(tensor_drawbox)

            if (augtype == 0):
                img2 = cv2.imread(img_lists[0] + "/" + img_lists[img_sequence[i]+1], 0) #for sequence augmentation
                img2 = cv2.resize(img2, dsize=(360, 240), interpolation=cv2.INTER_CUBIC)
                tensor_aug = self.transform(img2)
                anopatch = tensor_aug[:,posH:(posH+60), posW:(posW+60)] #crop patch
                tensor_image[:,posH:(posH+60), posW:(posW+60)] = anopatch #paste patch
            else: 
                k = random.randrange(0, 4)

                # print(tensor_image.shape)
                anopatch = tensor_image[:,posH:(posH+60), posW:(posW+60)] #ped2: 30, ped1:22, avenue:70
                anopatch = torch.rot90(anopatch, k, dims=(1,2))
                tensor_image[:,posH:(posH+60), posW:(posW+60)] = anopatch



            aug_box.append(tensor_image) #augmented frame cuboid

        #================pred gt===========================
        pred_img = cv2.imread(img_lists[0] + "/" + img_lists[self.num_of_img], 0)
        pred_img = cv2.resize(pred_img, dsize=(360, 240), interpolation=cv2.INTER_CUBIC) # ped2
        pred_image = self.transform(pred_img)
        pred_torch = torch.unsqueeze(pred_image, dim=0)
        #==================================================


        result = torch.cat(aug_box, dim=0) #augmented
        real = torch.cat(real_box, dim=0) #gt
        box = torch.cat(draw_box, dim=0)

        result = torch.unsqueeze(result, dim=0)
        real = torch.unsqueeze(real, dim=0)

        folder_name = (img_lists[0].split("/"))[-1]
        if self.mode == 'train':
            # return result, real, label.long() #recon
            return result, pred_torch, real, box
        else:
            # return real, real, folder_name #기본 rotation만 준 코드
            return real, pred_torch, folder_name #task5 (sequence shuffle 코드)


'''====================================='''
'''  10. sequence and rotation 4d - and '''
'''====================================='''
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
class VIDEO10(Dataset): #sequence shuffle
    def __init__(self, data_list='./train.txt', num_of_img=3, num_of_skip=1, mode='train'):
        with open(os.path.join(data_list)) as f:
            self.mode = mode
            self.img_list = []
            self.num_of_img = num_of_img
            self.num_of_skip = num_of_skip
            self.transform_forlabel = transforms.Compose([transforms.ToTensor()])
            self.transform = transforms.Compose([transforms.ToTensor()])
            # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

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
        # print(self.img_list)

        img_sequence = list(range(self.num_of_img-1))
        shuffle(img_sequence)

        posW = random.randrange(0, 10)
        posH = random.randrange(0, 60)

        augtype = random.randrange(0,2) #0이면 셔플, 1이면 로테이션

        for i in range(self.num_of_img-1): #pred의 경우 -1, recon의 경우 삭제해야함
            img = cv2.imread(img_lists[0] + "/" + img_lists[i+1], 0)
            img = cv2.resize(img, dsize=(540, 360), interpolation=cv2.INTER_CUBIC) # ped2
            # img = cv2.resize(img, dsize=(220, 154), interpolation=cv2.INTER_CUBIC) # ped1
            # img = cv2.resize(img, dsize=(630, 350), interpolation=cv2.INTER_CUBIC) # avn
            tensor_image = self.transform(img)
            tensor_for_gt = tensor_image.clone()
            tensor_image = torch.unsqueeze(tensor_image, dim=0)
            real_box.append(tensor_for_gt) #gt
            
            if self.mode == 'train':
                img_drawbox = img
                # print(img_drawbox.shape)
                img_drawbox[posH:(posH+60), posW:(posW+1)] = 255 #왼쪽 column
                img_drawbox[posH:(posH+60), (posW+60):(posW+61)] = 255 #오른쪽 column
                img_drawbox[posH:(posH+1), posW:(posW+60)] = 255 #위쪽 row
                img_drawbox[(posH+60):(posH+61), posW:(posW+61)] = 255 #아래쪽 row
                tensor_drawbox = self.transform(img_drawbox)
                draw_box.append(tensor_drawbox)
                print(augtype)
                if (augtype == 0): 
                    img2 = cv2.imread(img_lists[0] + "/" + img_lists[img_sequence[i]+1], 0) #for sequence augmentation
                    img2 = cv2.resize(img2, dsize=(540, 360), interpolation=cv2.INTER_CUBIC)
                    tensor_aug = self.transform(img2)
                    tensor_aug = torch.unsqueeze(tensor_aug, dim=0)
                    print(tensor_aug.shape)

                    anopatch = tensor_aug[:,:,posH:(posH+60), posW:(posW+60)] #crop patch
                    tensor_image[:,:,posH:(posH+60), posW:(posW+60)] = anopatch #paste patch
                else:
                    k = random.randrange(0, 4)
                    anopatch = tensor_image[:,:,posH:(posH+60), posW:(posW+60)] #ped2: 30, ped1:22, avenue:70
                    anopatch = torch.rot90(anopatch, k, dims=(2,3))
                    tensor_image[:,:,posH:(posH+60), posW:(posW+60)] = anopatch

                aug_box.append(tensor_image) #augmented frame cuboid

        #================pred gt===========================
        pred_img = cv2.imread(img_lists[0] + "/" + img_lists[self.num_of_img], 0)
        pred_img = cv2.resize(pred_img, dsize=(540, 360), interpolation=cv2.INTER_CUBIC) # ped2
        pred_image = self.transform(pred_img)
        pred_torch = torch.unsqueeze(pred_image, dim=0)
        #==================================================

        if self.mode == 'train':
            result = torch.cat(aug_box, dim=0) #augmented
            box = torch.cat(draw_box, dim=0)
        
        real = torch.cat(real_box, dim=0) #gt            
        real = torch.unsqueeze(real, dim=0)

        pred_torch = pred_torch.transpose(0,1)
        result = result.transpose(0,1)


        folder_name = (img_lists[0].split("/"))[-1]
        if self.mode == 'train':
            # return result, real, label.long() #recon
            return result, pred_torch, real, box
        else:
            # return real, real, folder_name #기본 rotation만 준 코드
            return real, pred_torch, folder_name #task5 (sequence shuffle 코드)


'''====================================='''
'''  11. sequence and rotation 4d - or random'''
'''====================================='''
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
class VIDEO11(Dataset): #sequence shuffle
    def __init__(self, data_list='./train.txt', num_of_img=3, num_of_skip=1, mode='train'):
        with open(os.path.join(data_list)) as f:
            self.mode = mode
            self.img_list = []
            self.num_of_img = num_of_img
            self.num_of_skip = num_of_skip
            self.transform_forlabel = transforms.Compose([transforms.ToTensor()])
            self.transform = transforms.Compose([transforms.ToTensor()])
            # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

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
        # print(self.img_list)

        img_sequence = list(range(self.num_of_img-1))
        shuffle(img_sequence)


        posW = random.randrange(0, 260)
        posH = random.randrange(30, 110)
        augtype = random.randrange(0,2) #0이면 tuvmf, 1이면 로테이션
        # augtype = 1
        k = random.randrange(0, 4)
        # print(augtype, k)

        for i in range(self.num_of_img-1): #pred의 경우 -1, recon의 경우 삭제해야함
            img = cv2.imread(img_lists[0] + "/" + img_lists[i+1], 0)
            img = cv2.resize(img, dsize=(360, 240), interpolation=cv2.INTER_CUBIC) # ped2
            tensor_image = self.transform(img)
            tensor_for_gt = tensor_image.clone()
            tensor_image = torch.unsqueeze(tensor_image, dim=0)
            real_box.append(tensor_for_gt) #gt
            
            if self.mode == 'train':
                img_drawbox = img
                # print(img_drawbox.shape)
                img_drawbox[posH:(posH+80), posW:(posW+1)] = 255 #왼쪽 column
                img_drawbox[posH:(posH+80), (posW+80):(posW+31)] = 255 #오른쪽 column
                img_drawbox[posH:(posH+1), posW:(posW+80)] = 255 #위쪽 row
                img_drawbox[(posH+80):(posH+81), posW:(posW+81)] = 255 #아래쪽 row
                tensor_drawbox = self.transform(img_drawbox)
                draw_box.append(tensor_drawbox)
                if (augtype == 0):
                    img2 = cv2.imread(img_lists[0] + "/" + img_lists[img_sequence[i]+1], 0) #for sequence augmentation
                    img2 = cv2.resize(img2, dsize=(360, 240), interpolation=cv2.INTER_CUBIC)
                    tensor_aug = self.transform(img2)
                    tensor_aug = torch.unsqueeze(tensor_aug, dim=0)
                    anopatch = tensor_aug[:,:,posH:(posH+90), posW:(posW+90)] #crop patch
                    tensor_image[:,:,posH:(posH+90), posW:(posW+90)] = anopatch #paste patch
                else: 
                    k = random.randrange(0, 4)
                    anopatch = tensor_image[:,:,posH:(posH+90), posW:(posW+90)] #ped2: 30, ped1:22, avenue:70
                    anopatch = torch.rot90(anopatch, k, dims=(2,3))
                    # print(tensor_image[:,:,posH:(posH+70), posW:(posW+70)].shape)
                    tensor_image[:,:,posH:(posH+90), posW:(posW+90)] = anopatch

                aug_box.append(tensor_image) #augmented frame cuboid


            # if self.mode == 'train':
            #     img_drawbox = img
            #     # print(img_drawbox.shape)
            #     img_drawbox[posH:(posH+60), posW:(posW+1)] = 255 #왼쪽 column
            #     img_drawbox[posH:(posH+60), (posW+60):(posW+61)] = 255 #오른쪽 column
            #     img_drawbox[posH:(posH+1), posW:(posW+60)] = 255 #위쪽 row
            #     img_drawbox[(posH+60):(posH+61), posW:(posW+61)] = 255 #아래쪽 row
            #     tensor_drawbox = self.transform(img_drawbox)
            #     draw_box.append(tensor_drawbox)

            #     img2 = cv2.imread(img_lists[0] + "/" + img_lists[img_sequence[i]+1], 0) #for sequence augmentation
            #     img2 = cv2.resize(img2, dsize=(360, 240), interpolation=cv2.INTER_CUBIC)
            #     tensor_aug = self.transform(img2)
            #     tensor_aug = torch.unsqueeze(tensor_aug, dim=0)
            #     anopatch = tensor_aug[:,:,posH:(posH+60), posW:(posW+60)] #crop patch
            #     anopatch = torch.rot90(anopatch, k, dims=(2,3))
            #     tensor_image[:,:,posH:(posH+60), posW:(posW+60)] = anopatch #paste patch


            #     aug_box.append(tensor_image) #augmented frame cuboid

        #================pred gt===========================
        pred_img = cv2.imread(img_lists[0] + "/" + img_lists[self.num_of_img], 0)
        pred_img = cv2.resize(pred_img, dsize=(360, 240), interpolation=cv2.INTER_CUBIC) # ped2
        pred_image = self.transform(pred_img)
        pred_torch = torch.unsqueeze(pred_image, dim=0)
        #==================================================

        if self.mode == 'train':
            result = torch.cat(aug_box, dim=0) #augmented
            box = torch.cat(draw_box, dim=0)
            result = result.transpose(0,1)
        
        real = torch.cat(real_box, dim=0) #gt            
        real = torch.unsqueeze(real, dim=0)

        pred_torch = pred_torch.transpose(0,1)
        # result = result.transpose(0,1)


        folder_name = (img_lists[0].split("/"))[-1]
        if self.mode == 'train':
            # return result, real, label.long() #recon
            return result, pred_torch, real, box
        else:
            # return real, real, folder_name #기본 rotation만 준 코드
            return real, pred_torch, folder_name #task5 (sequence shuffle 코드)

'''====================================='''
'''  12. 4d baseline'''
'''====================================='''
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
class VIDEO_baseline(Dataset): #sequence shuffle
    def __init__(self, data_list='./train.txt', num_of_img=3, num_of_skip=1, mode='train'):
        with open(os.path.join(data_list)) as f:
            self.mode = mode
            self.img_list = []
            self.num_of_img = num_of_img
            self.num_of_skip = num_of_skip
            self.transform_forlabel = transforms.Compose([transforms.ToTensor()])
            self.transform = transforms.Compose([transforms.ToTensor()])
            # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

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
        # print(self.img_list)

        img_sequence = list(range(self.num_of_img-1))
        shuffle(img_sequence)

        # posW = random.randrange(0, 300)
        # posH = random.randrange(60, 120)
        posW = random.randrange(0, 320)
        posH = random.randrange(30, 180)
        augtype = random.randrange(0,2) #0이면 tuvmf, 1이면 로테이션

        for i in range(self.num_of_img-1): #pred의 경우 -1, recon의 경우 삭제해야함
            img = cv2.imread(img_lists[0] + "/" + img_lists[i+1], 0)
            img = cv2.resize(img, dsize=(360, 240), interpolation=cv2.INTER_CUBIC) # ped2
            # img = cv2.resize(img, dsize=(220, 154), interpolation=cv2.INTER_CUBIC) # ped1
            # img = cv2.resize(img, dsize=(630, 350), interpolation=cv2.INTER_CUBIC) # avn
            tensor_image = self.transform(img)
            tensor_for_gt = tensor_image.clone()
            tensor_image = torch.unsqueeze(tensor_image, dim=0)
            real_box.append(tensor_for_gt) #gt
            
            if self.mode == 'train':
                img_drawbox = img
                # print(img_drawbox.shape)
                img_drawbox[posH:(posH+60), posW:(posW+1)] = 255 #왼쪽 column
                img_drawbox[posH:(posH+60), (posW+60):(posW+61)] = 255 #오른쪽 column
                img_drawbox[posH:(posH+1), posW:(posW+60)] = 255 #위쪽 row
                img_drawbox[(posH+60):(posH+61), posW:(posW+61)] = 255 #아래쪽 row
                tensor_drawbox = self.transform(img_drawbox)
                draw_box.append(tensor_drawbox)

        #================pred gt===========================
        pred_img = cv2.imread(img_lists[0] + "/" + img_lists[self.num_of_img], 0)
        pred_img = cv2.resize(pred_img, dsize=(360, 240), interpolation=cv2.INTER_CUBIC) # ped2
        pred_image = self.transform(pred_img)
        pred_torch = torch.unsqueeze(pred_image, dim=0)
        #==================================================

        if self.mode == 'train':
            result = torch.cat(aug_box, dim=0) #augmented
            # result = torch.unsqueeze(result, dim=0)
            box = torch.cat(draw_box, dim=0)
        
        real = torch.cat(real_box, dim=0) #gt            
        real = torch.unsqueeze(real, dim=0)
        real1 = real.copy()

        pred_torch = pred_torch.transpose(0,1)
        real1 = real.transpose(0,1)
        # result = result.transpose(0,1)


        folder_name = (img_lists[0].split("/"))[-1]
        if self.mode == 'train':
            # return result, real, label.long() #recon
            return real1, pred_torch, real, box
        else:
            # return real, real, folder_name #기본 rotation만 준 코드
            return real, pred_torch, folder_name #task5 (sequence shuffle 코드)



class Label_loader:
    def __init__(self, dataset, video_folders):
        self.data_root ='/SSD1/chaewon/datasets/'
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



if __name__ == "__main__":
    dataset = VIDEO(data_list='./ped2_test.txt', num_of_img=4, num_of_skip=1)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    for b_idx, data in enumerate(train_loader):
        img, label = data

        print(img.shape, label.shape)
        # print(label)
        break