import sys
import os
import datetime
import numpy as np
import time
import torch
import torchvision
from torch.utils.data import DataLoader
import cv2
import io
from tqdm import tqdm
from sklearn import metrics
from dataloader import Label_loader, FrameCount
from math import log10, sqrt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import torchvision.transforms as transforms

def evaluation(test_loader, model1, dataset, video):
    total_frame = FrameCount(dataset)
    print('Total test frames: %d' % total_frame)
    task = tqdm(test_loader)
    score_group = []
    score=[]
    scores = np.array([], dtype=np.float64)
    labels = np.array([], dtype=np.int8)
    num=1
    criterionL2 = torch.nn.MSELoss()   
    video_folders = os.listdir("./data/%s/testing/frames/"%(dataset))
    video_folders.sort()
    video_folders = [os.path.join("./data/%s/testing/frames/"%(dataset), vid) for vid in video_folders]

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start = time.time()
    starter.record()

    with torch.no_grad():
        for i, data in enumerate(task):
            aug_img, real_img, clip_name = data
            aug_img, real_img = aug_img.cuda(), real_img.cuda()
            fake_img= model1(aug_img)
            clip_name = int(clip_name[0])
            for i in range(fake_img.shape[0]): 
                if clip_name > num: 
                    score_group.append(np.array(score))
                    num = clip_name
                    score = []
                loss_G_L2 = criterionL2(fake_img[i], real_img[i]) 
                Psnr = log10(1.0 / sqrt(loss_G_L2)) 
                correct_score = Psnr
            score.append(float(correct_score))
    score_group.append(np.array(score))   

    print('All frames were detected, begin to compute AUC.')
    gt_loader = Label_loader(dataset=dataset, video_folders=video_folders)  
    gt = gt_loader()
    assert len(score_group) == len(gt), f'Ground truth has {len(gt)} videos, but got {len(score_group)} detected videos.'

    for i in range(len(score_group)):
            distance = score_group[i]
            max_d = max(distance)
            min_d = min(distance)
            distance -= min_d  
            distance /= (max_d -min_d)
            scores = np.concatenate((scores, distance[:]), axis=0) 
            labels = np.concatenate((labels, gt[i][5:]), axis=0) 
    assert scores.shape == labels.shape, \
        f'Ground truth has {labels.shape[0]} frames, but got {scores.shape[0]} detected frames.'
    try:
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0) 
        auc = metrics.auc(fpr, tpr)       
        ender.record() 
        torch.cuda.synchronize()
        inference_time = starter.elapsed_time(ender)*1e-3 
        print("FPS:", total_frame/inference_time) 
    except: print("error")
    return auc         
