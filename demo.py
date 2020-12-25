import cv2 as cv
import sys
sys.path.append('registration')
import math
from RetinaFace.newdetector import RetinafaceDetector
import torch
import numpy as np
from face_network.Network import InceptionResnetV1


def process(path, output_size):
    image = cv.imread(path)
    _, facial5points = detector.detect_faces(image)
    facial5points = np.reshape(facial5points[0], (2, 5))
    theta = math.atan((facial5points[1][1] - facial5points[1][0]) / (facial5points[0][1] - facial5points[0][0])) * 180 / math.pi
    matRotate = cv.getRotationMatrix2D((0, 0), theta, 0.80)
    image_affined = cv.warpAffine(image, matRotate, (1280, 720))
    try:
        image_cropped = detector.extract_faces(image_affined, output_size=(output_size, output_size), is_max=True)
    except:
        image_cropped = detector.extract_faces(image, output_size=(output_size, output_size), is_max=True)
    output = np.array([image_cropped[:, :, 2], image_cropped[:, :, 1], image_cropped[:, :, 0]])
    output = torch.Tensor(output).to(device)
    output = (output - 127.5) / 128
    return output


def test():
    path = 'data/1/meter_0.5/ZED_Left_2020_07_31_14_11_01_984.0.jpg'
    img1_cropped = process(path, 160)
    face_embedding = resnet(img1_cropped.unsqueeze(0)).detach().cpu()
    return face_embedding


if __name__ == "__main__":
    model_dir = 'face_network/inceptionNet.pth'
    checkpoint = torch.load(model_dir)
    detector = RetinafaceDetector()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    resnet = resnet.eval()
    resnet.load_state_dict(checkpoint['inceptionNet'])
    face_embedding = test()
    print(face_embedding)
