"""
File Name:    crossdist_test.py
Author:       Xinyu Li
Email:        3170103467@zju.edu.cn
Time:         2020/9/31
function:     cross distance comparison and draw ROC curves
             (one is frontal face, the other is frontal face or side face)
"""
import cv2 as cv
import matplotlib.pyplot as plt
import torch
import numpy as np
import os


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
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dists.size
    return tpr, fpr, acc


def test2(face_idx0,distance_idx1,face_idx1):
    rootdir = "./newdata/"
    actual_issame = []
    dists = []
    genuine_number = 0
    imposter_number = 0
    # feature = dict_feature()
    feature= np.load('feature_dict.npy',allow_pickle=True).item()
    dir_lists = []
    for line in open("dir.txt", "r"):
        dir_list = line.rstrip('\n')
        dir_lists.append(dir_list)
    # print(dir_lists)
    # generate genuine pairs
    for folder_idx in range(1, NumberOfFiles+1):
        frontal_faces0 = []
        side_faces1 = []
        frontal_faces1 = []
        directory_dir1 = os.path.join(rootdir, str(folder_idx) +  face_idx0)
        directory_dir2 = os.path.join(rootdir, str(folder_idx) + distance_idx1)
        directory_dir3 =  os.path.join(rootdir, str(folder_idx) + face_idx1)
        for dir_list in dir_lists:
            if directory_dir1 in dir_list:
                frontal_faces0.append(dir_list)
            elif directory_dir2 in dir_list:
                side_faces1.append(dir_list)
            elif directory_dir3 in dir_list:
                frontal_faces1.append(dir_list)
            else:
                pass
        for i in range(len(frontal_faces0)):
            # path = os.path.join(directory_dir1, frontal_faces0[i])
            path = frontal_faces0[i]
            print('test first image ', path)
            face_embedding1 =feature[path]
            for i in range(len(side_faces1)):
                path =  side_faces1[i]
                print(path)
                face_embedding2 = feature[path]
                dist = (face_embedding1 - face_embedding2).norm().item()
                # print ('euclidean distance ', path, dist)
                dists.append(dist)
                actual_issame.append(True)
                genuine_number += 1
            for i in range(len(frontal_faces1)):
                path = frontal_faces1[i]
                face_embedding2 =feature[path]
                dist = (face_embedding1 - face_embedding2).norm().item()
                # print ('euclidean distance ', path, dist)
                dists.append(dist)
                actual_issame.append(True)
                genuine_number += 1

    # generate imposter pairs
    for folder_idx in range(1, NumberOfFiles+1):
        frontal_faces0= []
        directory_dir1 = os.path.join(rootdir, str(folder_idx) + face_idx0)
        for dir_list in dir_lists:
            if directory_dir1 in dir_list:
                frontal_faces0.append(dir_list)
        for i in range(len(frontal_faces0)):
            path = frontal_faces0[i]
            print('folder{},first test image'.format(folder_idx), path)
            face_embedding1 = feature[path]
            for idx in range(1, NumberOfFiles+1):
                if idx == folder_idx:
                    continue
                print('folder:',idx)
                side_faces1 = []
                directory_dir2 = os.path.join(rootdir, str(idx) + distance_idx1)
                for dir_list in dir_lists:
                    if directory_dir2 in dir_list:
                        side_faces1.append(dir_list)
                for i in range(len(side_faces1)):
                    path = side_faces1[i]
                    face_embedding2 = feature[path]
                    dist = (face_embedding1 - face_embedding2).norm().item()
                    # print ('euclidean distance ', path, dist)
                    dists.append(dist)
                    actual_issame.append(False)
                    imposter_number += 1
                directory_dir3 = os.path.join(rootdir, str(idx) + face_idx1)
                frontal_faces1 = []
                if directory_dir3 in dir_list:
                    frontal_faces1.append(dir_list)
                for i in range(len( frontal_faces1)):
                    path =  frontal_faces1[i]
                    face_embedding2 = feature[path]
                    dist = (face_embedding1 - face_embedding2).norm().item()
                    # print ('euclidean distance ', path, dist)
                    dists.append(dist)
                    actual_issame.append(False)
                    imposter_number += 1

    print('genuine_number', genuine_number)
    print('imposter number', imposter_number)
    # draw ROC
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
    plt.semilogx()  # sets the X axis to the logarithmic axis
    plt.xlim(0, 1)
    plt.show()
    # calculate eer
    delta = 0.001
    err = 1
    best_threshold = 0
    for i in range(len(thresholds)):
        if (abs(fprs[i] - (1 - tprs[i])) < delta):
            if err > fprs[i]:
                err = fprs[i]
                best_threshold = thresholds[i]
    print('equal error rate is', err)
    print('threshold is', best_threshold)

if __name__ == "__main__":
    NumberOfFiles = 18
    test2(face_idx0='/0.5frontal/', distance_idx1='/meter_1/', face_idx1='/1.0frontal/')   # cross distance comparison
    # we can use the following selections
    '''
        face_idx0 = '/0.5frontal/', distance_idx1 = '/meter_1/', face_idx1 = '/1.0frontal/'
        face_idx0 = '/0.5frontal/', distance_idx1 = '/meter_1.5/', face_idx1 = '/1.5frontal/'
        face_idx0 = '/1frontal/', distance_idx1 = '/meter_1.5/', face_idx1 = '/1.5frontal/'
    '''

