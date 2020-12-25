#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Name:    samedist_test.py
Author:       Xinyu Li
Email:        3170103467@zju.edu.cn
Time:         2020/9/31
function:     compare pairs in same distance and draw ROC curves
             (one is frontal face, the other is frontal face or side face)
"""
import cv2 as cv
import numpy as np
import torch
import sys
sys.path.append('..')
sys.path.append('...')
from face_network.Network import InceptionResnetV1
import os


def dict_feature():
    rootdir = "./newdata/"
    feature = {}
    # generate dictionary to store feature vector. {image path : deature vector}
    for folder_idx in range(1,19):
        print(folder_idx)
        for idx in ['/meter_0.5/','/meter_1/','/meter_1.5/','/0.5frontal/','/1.0frontal/','/1.5frontal/']:
            directory_dir = os.path.join(rootdir, str(folder_idx) + idx)
            list_file = os.listdir(directory_dir)
            for i in range(len(list_file)):
                img1_path = os.path.join(directory_dir, list_file[i])
                raw = cv.imread(img1_path)
                img1_cropped = np.array([raw[:, :, 2], raw[:, :, 1], raw[:, :, 0]])
                img1_cropped = torch.Tensor(img1_cropped).to(device)
                img1_cropped = (img1_cropped - 127.5) / 128
                face_embedding = resnet(img1_cropped.unsqueeze(0)).detach().cpu()
                feature[img1_path] = face_embedding
    np.save('feature_dict.npy', feature)
    return feature


if __name__ == '__main__':
    (p,_) = os.path.split(__file__)
    p = p[:-23]
    model_dir = p + 'face_network/inceptionNet.pth'
    checkpoint = torch.load(model_dir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet = resnet.eval()
    resnet.load_state_dict(checkpoint['inceptionNet'])
    NumberOfFiles = 18
    dict_feature()
