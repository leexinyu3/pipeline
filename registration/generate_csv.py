"""
File Name:    generate_csv.py
Author:       Xinyu Li
Email:        3170103467@zju.edu.cn
Time:         2020/10/10
function:     write 5 facial landmarks of 160 size image to csv file.
"""
import cv2 as cv
import sys
sys.path.append('registration')
from RetinaFace.newdetector import RetinafaceDetector
import torch
import os
import csv
import numpy as np
import math
# return aligned dataset of same size


def aligned_face(img2,output_size = (160,160)):
    _, facial5points = detector.detect_faces(img2)
    facial5points = np.reshape(facial5points[0], (2, 5))
    theta = math.atan((facial5points[1][1] - facial5points[1][0]) / (
            facial5points[0][1] - facial5points[0][0])) * 180 / math.pi
    matRotate = cv.getRotationMatrix2D((0, 0), theta, 0.80)
    IMG2 = cv.warpAffine(img2, matRotate, (1280, 720))
    try:
        img2_cropped = detector.extract_faces(IMG2,output_size = output_size,is_max=True)
    except:
        img2_cropped = detector.extract_faces(img2,output_size = output_size,is_max=True)
    return img2_cropped

def write_landmarks(output_croppeddata = False,output_size = (160,160)):
    identity_number = 18
    rootdir = "../data/"
    targetdir ='../cropDataset160/'
    distance_idx = '/meter_0.5/'
    # distances_idx = ['/meter_0.5/','/meter_1/','/meter_1.5/']
    csv_f = open('./facial5points.CSV', 'a', encoding='utf-8', newline='')
    # for distance_idx in distances_idx:
    for folder_idx in range(1,identity_number+1):
        print(folder_idx)
        directory_dir = os.path.join(rootdir, str(folder_idx) + distance_idx)
        list_file = os.listdir(directory_dir)
        for i in range(len(list_file)):
            if list_file[i].endswith('.tiff'):
                continue
            path = os.path.join(directory_dir, list_file[i])
            print(path)
            img2 = cv.imread(path)
            img2_cropped = aligned_face(img2,output_size = output_size)
            _, facial5points = detector.detect_faces(img2_cropped)
            facial5points = np.reshape(facial5points[0], (2, 5))

            # output cropped dataset
            if(output_croppeddata):
                img_path = os.path.join(targetdir, str(folder_idx) + distance_idx )
                if not os.path.exists(img_path):
                    os.makedirs(img_path)
                cv.imwrite(img_path + list_file[i], img2_cropped)
            # write five facial points to csv file
            try:
                csv_writer = csv.writer(csv_f)
                csv_writer.writerow([str(folder_idx) + distance_idx+list_file[i],str(facial5points[0][0]),str(facial5points[0][1]),str(facial5points[0][2]),str(facial5points[0][3]),str(facial5points[0][4]),
                                     str(facial5points[1][0]),str(facial5points[1][1]),str(facial5points[1][2]),str(facial5points[1][3]),str(facial5points[1][4])])
                # landmark.append([str(folder_idx) + distance_idx + list_file[i], str(facial5points[0][0]), str(facial5points[0][1]),
                #  str(facial5points[0][2]), str(facial5points[0][3]), str(facial5points[0][4]),
                #  str(facial5points[1][0]), str(facial5points[1][1]), str(facial5points[1][2]), str(facial5points[1][3]),
                #  str(facial5points[1][4])])
            except:
                csv_writer = csv.writer(csv_f)
                csv_writer.writerow([str(folder_idx) + distance_idx+list_file[i]])
                print('error!',path)
    csv_f.close()

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    detector = RetinafaceDetector()
    write_landmarks(output_croppeddata = False,output_size = (160,160))

