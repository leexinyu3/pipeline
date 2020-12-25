"""
File Name:    registration.py
Author:       Xinyu Li
Email:        3170103467@zju.edu.cn
Time:         2020/10/10
function:     select 5 or 6 images from meter_0.5 of each person including 1 or 2 frontal faces and other faces of different pose, and put then in a new folder called '/meter_0.5_selected'
"""

import csv
from similarity import poseSimilarity
import os
import shutil
import random


def register_face():
    rootdir = "../data/"
    targetdir =  "../data/"
    distance_idx = '/meter_0.5/'
    output_size = 160
    # if you want to change the size, you need to run generate_csv.py again to generate the new facial5points.CSV by changing the parameter 'output_size' in function 'write_landmarks',eg:write_landmarks(output_size= (128,128))
    threshold = 1.2 #you can change
    dir_list = []
    frontal_faces = {}
    for i in range(1, 19):
        frontal_faces[i] = []
    with open('facial5points.csv', 'r') as f:
        reader = csv.reader(f)
        reader = list(reader)
    for i in range(len(reader)):
        dir_list.append(reader[i][0])
        if output_size / 2 - 10 < float(reader[i][3]) < output_size / 2 + 10:
            identity = int(reader[i][0].split('/')[0])
            frontal_faces[identity].append(reader[i][0])
    for i in range(1, 19):
        path = os.path.join(targetdir, str(i) + '/meter_0.5_selected/')
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            shutil.rmtree(path)
            os.makedirs(path)
    for idx in range(1, 19):
        image_list = []
        directory_dir1 = os.path.join(rootdir, str(idx) + distance_idx)
        faces = os.listdir(directory_dir1)
        random.shuffle(frontal_faces[idx])
        if len(frontal_faces[idx]) > 1:
            image_list.append(frontal_faces[idx][0])
            # image_list.append(frontal_faces[idx][1])#if cancelled this comment, we selected 2 frontal faces
        random.shuffle(faces)
        for i in range(len(faces)):
            if len(image_list) > 5:
                break
            if faces[i].endswith('.tiff'):
                continue
            flag = 1
            path2 = os.path.join(str(idx) + distance_idx, faces[i])
            index2 = dir_list.index(path2)
            lenth = len(image_list)
            for i in range(lenth):
                index1 = dir_list.index(image_list[i])
                pose_simiarity = poseSimilarity(reader[index1], reader[index2])
                if pose_simiarity < threshold:
                    flag = 0
            if flag:
                image_list.append(path2)
        for img in image_list:
            path = os.path.join(rootdir, img)
            new_path = os.path.join(targetdir, img).replace('meter_0.5', 'meter_0.5_selected')
            shutil.copy(path, new_path)


if __name__ == "__main__":
    register_face()
    rootdir = "../data/"
    targetdir =  "../data/"
    for i in range(1,19):
        path = os.path.join(targetdir, str(i) + '/meter_0.5_selected/')
        faces = os.listdir(path)
        print('人脸个数：',len(faces))


