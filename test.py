#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/4 9:43
# @Author  : Shark
# @Site    : 
# @File    : test.py.py
# @Software: PyCharm


from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

"""
File Name:    minidist_test.py
Author:       Xinyu Li
Email:        3170103467@zju.edu.cn
Time:         2020/10/10
function:     compare test images from 1/1.5 folder with enrolled images in meter_0.5_selected folder.
"""
import cv2 as cv
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import sys

sys.path.append('..')
sys.path.append('...')


def calculate_accuracy(threshold, dists, actual_issame):
    dists = np.array(dists)
    predict_issame = np.less(dists, threshold)
    # What is judged to be the same person is actually the same person
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    # What is judged to be the same person is actually not the same person
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    # What is judged to be the differnet person is actually not the same person
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    # What is judged to be the differnet person is actually the same person
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dists.size
    return tpr, fpr, acc
def dict_feature():
    NumberOfFiles = 18
    rootdir = "./data/"
    feature = {}
    # generate dictionary
    for folder_idx in range(1, NumberOfFiles+1):
        print(folder_idx)
        for idx in ['/meter_0.5/','/meter_1/','/meter_1.5/']:
            directory_dir = os.path.join(rootdir, str(folder_idx) + idx)
            list_file = os.listdir(directory_dir)
            for i in range(len(list_file)):
                if list_file[i].endswith('.tiff'):
                    continue
                path = os.path.join(directory_dir, list_file[i])
                img1 = Image.open(path)
                # Get cropped and prewhitened image tensor
                img_cropped1 = mtcnn(img1)
                if img_cropped1 is None:
                    continue
                # Calculate embedding (unsqueeze to add batch dimension)
                face_embedding = resnet(img_cropped1.to(device).unsqueeze(0)).detach().cpu()
                feature[path] = face_embedding

    np.save('MTCNN_featuredict.npy', feature)
    return feature


def test():
    rootdir = "./data/"
    actual_issame = []
    dists = []
    genuine_number = 0
    imposter_number = 0

    # feature = dict_feature()
    feature = np.load('MTCNN_featuredict.npy', allow_pickle=True).item()
    distance_idx0 = '/meter_0.5_selected/'  # this folder includes 5 or 6 images we selected
    distance_idx1 = '/meter_1/'
    distance_idx2 = '/meter_1.5/'

    # generate genuine pairs
    for folder_idx in range(1, NumberOfFiles + 1):

        directory_dir1 = os.path.join(rootdir, str(folder_idx) + distance_idx0)
        directory_dir2 = os.path.join(rootdir, str(folder_idx) + distance_idx1)
        directory_dir3 = os.path.join(rootdir, str(folder_idx) + distance_idx2)
        faces0 = os.listdir(directory_dir1)
        faces1 = os.listdir(directory_dir2)
        faces2 = os.listdir(directory_dir3)

        for i in range(len(faces1)):
            if faces1[i].endswith('.tiff'):
                continue
            best_dist = 100
            path = os.path.join(directory_dir2, faces1[i]).replace('meter_0.5_selected', 'meter_0.5')
            print('test image', path)

            # img1 = Image.open(path)
            # # Get cropped and prewhitened image tensor
            # img_cropped1 = mtcnn(img1).to(device)
            # # Calculate embedding (unsqueeze to add batch dimension)
            # face_embedding1 = resnet(img_cropped1.unsqueeze(0)).detach().cpu()

            if path not in feature.keys():  # 或dict1.keys()
                continue
            face_embedding1 = feature[path]

            for i in range(len(faces0)):
                if faces0[i].endswith('.tiff'):
                    continue
                path = os.path.join(directory_dir1, faces0[i]).replace('meter_0.5_selected', 'meter_0.5')

                # img2 = Image.open(path)
                # # Get cropped and prewhitened image tensor
                # img_cropped2 = mtcnn(img2).to(device)
                # # Calculate embedding (unsqueeze to add batch dimension)
                # face_embedding2 = resnet(img_cropped2.unsqueeze(0)).detach().cpu()
                if path not in feature.keys():  # 或dict1.keys()
                    continue
                face_embedding2 = feature[path]
                dist = (face_embedding1 - face_embedding2).norm().item()
                if dist < best_dist:
                    best_dist = dist
                    best_path = path
            dists.append(best_dist)
            print(best_path, best_dist)
            actual_issame.append(True)
            genuine_number += 1

        for i in range(len(faces2)):
            if faces2[i].endswith('.tiff'):
                continue
            best_dist = 100
            path = os.path.join(directory_dir3, faces2[i]).replace('meter_0.5_selected', 'meter_0.5')
            print('test image', path)

            # img1 = Image.open(path)
            #     # Get cropped and prewhitened image tensor
            # img_cropped1 = mtcnn(img1).to(device)
            #
            #     # Calculate embedding (unsqueeze to add batch dimension)
            # face_embedding1 = resnet(img_cropped1.unsqueeze(0)).detach().cpu()
            if path not in feature.keys():  # 或dict1.keys()
                continue
            face_embedding1 = feature[path]

            for i in range(len(faces0)):
                if faces0[i].endswith('.tiff'):
                    continue
                path = os.path.join(directory_dir1, faces0[i]).replace('meter_0.5_selected', 'meter_0.5')
                # img2 = Image.open(path)
                # # Get cropped and prewhitened image tensor
                # img_cropped2 = mtcnn(img2).to(device)
                #
                # # Calculate embedding (unsqueeze to add batch dimension)
                # face_embedding2 = resnet(img_cropped2.unsqueeze(0)).detach().cpu()
                if path not in feature.keys():  # 或dict1.keys()
                    continue
                face_embedding2 = feature[path]
                dist = (face_embedding1 - face_embedding2).norm().item()

                if dist < best_dist:
                    best_dist = dist
                    best_path = path
            dists.append(best_dist)
            print(best_path, best_dist)
            actual_issame.append(True)
            genuine_number += 1
    # generate imposter pairs

    for folder_idx in range(1, NumberOfFiles + 1):
        directory_dir2 = os.path.join(rootdir, str(folder_idx) + distance_idx1)
        faces1 = os.listdir(directory_dir2)
        for i in range(len(faces1)):
            if faces1[i].endswith('.tiff'):
                continue
            path = os.path.join(directory_dir2, faces1[i]).replace('meter_0.5_selected', 'meter_0.5')
            print('test image:', path)
            # img1 = Image.open(path)
            # img_cropped1 = mtcnn(img1).to(device)
            # # Calculate embedding (unsqueeze to add batch dimension)
            # face_embedding1 = resnet(img_cropped1.unsqueeze(0)).detach().cpu()
            if path not in feature.keys():  # 或dict1.keys()
                continue
            face_embedding1 = feature[path]

            for idx in range(1, NumberOfFiles + 1):
                best_dist = 100
                if idx == folder_idx:
                    continue
                print('folder:', idx)
                directory_dir1 = os.path.join(rootdir, str(idx) + distance_idx0)
                faces0 = os.listdir(directory_dir1)
                for i in range(len(faces0)):
                    if faces0[i].endswith('.tiff'):
                        continue
                    path = os.path.join(directory_dir1, faces0[i]).replace('meter_0.5_selected', 'meter_0.5')
            #         img0 = Image.open(path)
            #         img_cropped0 = mtcnn(img0).to(device)
            #
            # # Calculate embedding (unsqueeze to add batch dimension)
            #         face_embedding2 = resnet(img_cropped0.unsqueeze(0)).detach().cpu()

                    if path not in feature.keys():  # 或dict1.keys()
                        continue
                    face_embedding2 = feature[path]
                    dist = (face_embedding1 - face_embedding2).norm().item()

                    if dist < best_dist:
                        best_dist = dist
                        best_path = path
                dists.append(best_dist)
                print(best_path, best_dist)
                actual_issame.append(False)
                imposter_number += 1
        directory_dir3 = os.path.join(rootdir, str(folder_idx) + distance_idx2)
        faces2 = os.listdir(directory_dir3)
        for i in range(len(faces2)):
            if faces2[i].endswith('.tiff'):
                continue
            path = os.path.join(directory_dir3, faces2[i]).replace('meter_0.5_selected', 'meter_0.5')
            print('test image:', path)
            # img1 = Image.open(path)
            # img_cropped1 = mtcnn(img1).to(device)
            #
            # # Calculate embedding (unsqueeze to add batch dimension)
            # face_embedding1 = resnet(img_cropped1.unsqueeze(0)).detach().cpu()
            if path not in feature.keys():  # 或dict1.keys()
                continue
            face_embedding1 = feature[path]
    for idx in range(1, NumberOfFiles + 1):
                best_dist = 100
                if idx == folder_idx:
                    continue
                print('folder:', idx)
                directory_dir1 = os.path.join(rootdir, str(idx) + distance_idx0)
                faces0 = os.listdir(directory_dir1)
                for i in range(len(faces0)):
                    if faces0[i].endswith('.tiff'):
                        continue
                    path = os.path.join(directory_dir1, faces0[i]).replace('meter_0.5_selected', 'meter_0.5')
                    # img0 = Image.open(path)
                    # img_cropped0 = mtcnn(img0).to(device)
                    # # Calculate embedding (unsqueeze to add batch dimension)
                    # face_embedding2 = resnet(img_cropped0.unsqueeze(0)).detach().cpu()
                    if path not in feature.keys():  # 或dict1.keys()
                        continue
                    face_embedding2 = feature[path]

                    dist = (face_embedding1 - face_embedding2).norm().item()
                    if dist < best_dist:
                        best_dist = dist
                        best_path = path
                dists.append(best_dist)
                print(best_path, best_dist)
                actual_issame.append(False)
                imposter_number += 1

    print('genuine_number', genuine_number)
    print('imposter number', imposter_number)

    tprs = []
    fprs = []
    thresholds = np.arange(0, 2, 0.001)

    for idx, threshold in enumerate(thresholds):
        tpr, fpr, acc = calculate_accuracy(threshold, dists, actual_issame)
        tprs.append(tpr)
        fprs.append(fpr)
    tprs = np.array(tprs)
    fprs = np.array(fprs)
    plt.xlabel('FAR')
    plt.ylabel('GAR')
    plt.title('ROC curve')
    plt.plot(fprs, tprs)
    plt.semilogx()  # Sets the X axis to the logarithmic axis
    plt.xlim(0.0001, 1)
    plt.show()

    delta = 0.001
    err = 1
    for i in range(len(thresholds)):
        if (abs(fprs[i] - (1 - tprs[i])) < delta):
            if err > fprs[i]:
                err = fprs[i]
    print('equal error rate is', err)


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN( image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709,device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    # dict_feature()
    NumberOfFiles = 18
    test()


