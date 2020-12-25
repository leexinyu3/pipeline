import csv
from numpy import *
import math

def poseSimilarity(img1,img2):
    for i in range(1,11):
        img1[i] = float(img1[i])
        img2[i] = float(img2[i])
    length1 = math.sqrt((img1[1]-img1[3])**2+(img1[6]-img1[8])**2)
    length2 = math.sqrt((img1[3]-img1[4])**2+(img1[8]-img1[9])**2)
    length3 = math.sqrt((img1[1]-img1[4])**2+(img1[6]-img1[9])**2)
    length4 = math.sqrt((img1[2]-img1[3])**2+(img1[7]-img1[8])**2)
    length5 = math.sqrt((img1[3]-img1[5])**2+(img1[8]-img1[10])**2)
    length6 = math.sqrt((img1[2]-img1[5])**2+(img1[7]-img1[10])**2)
    length7 = math.sqrt((img1[1] - img1[2]) ** 2 + (img1[6] - img1[7]) ** 2)
    length8 = math.sqrt((img1[4] - img1[5]) ** 2 + (img1[9] - img1[10]) ** 2)
    theta1 = math.acos((length1 ** 2 + length7 ** 2 - length4 ** 2) / (2 * length1 * length7))
    theta2 = math.acos((length4 ** 2 + length7 ** 2 - length1 ** 2) / (2 * length4 * length7))
    theta3 = math.acos((length4 ** 2 + length6 ** 2 - length5 ** 2) / (2 * length4 * length6))
    theta4 = math.acos((length5 ** 2 + length6 ** 2 - length4 ** 2) / (2 * length5 * length6))
    theta5 = math.acos((length5 ** 2 + length8 ** 2 - length2 ** 2) / (2 * length5 * length8))
    theta6 = math.acos((length2 ** 2 + length8 ** 2 - length5 ** 2) / (2 * length2 * length8))
    theta7 = math.acos((length2 ** 2 + length3 ** 2 - length1 ** 2) / (2 * length2 * length3))
    theta8 = math.acos((length1 ** 2 + length3 ** 2 - length2 ** 2) / (2 * length1 * length3))
    length1 = math.sqrt((img2[1]-img2[3])**2+(img2[6]-img2[8])**2)
    length2 = math.sqrt((img2[3]-img2[4])**2+(img2[8]-img2[9])**2)
    length3 = math.sqrt((img2[1]-img2[4])**2+(img2[6]-img2[9])**2)
    length4 = math.sqrt((img2[2]-img2[3])**2+(img2[7]-img2[8])**2)
    length5 = math.sqrt((img2[3]-img2[5])**2+(img2[8]-img2[10])**2)
    length6 = math.sqrt((img2[2]-img2[5])**2+(img2[7]-img2[10])**2)
    length7 = math.sqrt((img2[1] - img2[2]) ** 2 + (img2[6] - img2[7]) ** 2)
    length8 = math.sqrt((img2[4] - img2[5]) ** 2 + (img2[9] - img2[10]) ** 2)
    phi1 = math.acos((length1 ** 2 + length7 ** 2 - length4 ** 2) / (2 * length1 * length7))
    phi2 = math.acos((length4 ** 2 + length7 ** 2 - length1 ** 2) / (2 * length4 * length7))
    phi3 = math.acos((length4 ** 2 + length6 ** 2 - length5 ** 2) / (2 * length4 * length6))
    phi4 = math.acos((length5 ** 2 + length6 ** 2 - length4 ** 2) / (2 * length5 * length6))
    phi5 = math.acos((length5 ** 2 + length8 ** 2 - length2 ** 2) / (2 * length5 * length8))
    phi6 = math.acos((length2 ** 2 + length8 ** 2 - length5 ** 2) / (2 * length2 * length8))
    phi7 = math.acos((length2 ** 2 + length3 ** 2 - length1 ** 2) / (2 * length2 * length3))
    phi8 = math.acos((length1 ** 2 + length3 ** 2 - length2 ** 2) / (2 * length1 * length3))
    similarity = abs(theta1-phi1) + abs(theta2-phi2) + abs(theta3-phi3) + abs(theta4-phi4) + abs(theta5-phi5) + abs(theta6-phi6) + abs(theta7-phi7) + abs(theta8-phi8)
    return similarity

if __name__ == "__main__":


    path1 = '1/meter_0.5/ZED_Right_2020_07_31_14_11_02_662.0.jpg'
    path2 = '1/meter_0.5/ZED_Right_2020_07_31_14_11_01_984.0.jpg'

    print(path2)
    with open('facial5points.csv', 'r') as f:
        reader = csv.reader(f)
        reader = list(reader)
        dir_list = []
        for i in range(len(reader)):
            dir_list.append(reader[i][0])
    index1 = dir_list.index(path1)
    index2 = dir_list.index(path2)

    pose = poseSimilarity(reader[index1],reader[index2])
    print(pose)



