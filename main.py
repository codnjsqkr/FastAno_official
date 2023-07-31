import sys
import os
import argparse
import torch
from torch.utils.data import DataLoader
import dataloader as ds 
from tqdm import tqdm
from fastanoAE import *
from inference import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='dataset', default='ped2')
    parser.add_argument('-t', '--task', help='task to perform: train/test', default='test')
    args = vars(parser.parse_args())
    dataset = args['dataset']
    task = str(args['task'])
    print('Selected task is %s' % task)
    size_of_batch = 1
    test_dataset = ds.VIDEO11(data_list='./%s_test.txt'%dataset, num_of_img=6, num_of_skip=1, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=size_of_batch, shuffle=False, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Generator(size_of_batch, 1)
    model.cuda()
    g_path = './weights/ped2_best.pth'
    model.load_state_dict(torch.load(g_path), strict=False)
    model.eval()                
    print("Start!")
    print("------------------------")
    print("Dataset:",dataset)
    auc = evaluation(test_loader, model, dataset, video=False)
    print("AUC is ", auc)
    print("------------------------")

if __name__ == '__main__':
    main(sys.argv)