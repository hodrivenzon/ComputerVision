import cv2
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# from utils import label_map_util
#
# from utils import visualization_utils as vis_util
# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades



def webcam_dilation():
    cap = cv2.VideoCapture(0)
    while (True):
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([30, 150, 50])
        upper_red = np.array([255, 255, 180])

        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(frame, frame, mask=mask)

        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(mask, kernel, iterations=1)
        dilation = cv2.dilate(mask, kernel, iterations=1)

        cv2.imshow('Original', frame)
        cv2.imshow('Mask', mask)
        cv2.imshow('Erosion', erosion)
        cv2.imshow('Dilation', dilation)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def save_video():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(r'C:\Users\hodda\Documents\output\output.avi', fourcc, 20.0, (640, 480))

    while (True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.write(frame)
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def face_recognition():
    # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')
    # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
    # eye_cascade = cv2.CascadeClassifier('parojos.xml')
    eye_cascade = cv2.CascadeClassifier('parojosG.xml')


    cap = cv2.VideoCapture(0)
    while (True):
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def adding_drawing_on_face():
    # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')
    # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
    # eye_cascade = cv2.CascadeClassifier('parojos.xml')
    eye_cascade = cv2.CascadeClassifier('parojosG.xml')


    cap = cv2.VideoCapture(0)
    while (True):
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # dst = cv2.imread(r"C:\Users\hodda\PycharmProjects\Bootcamp\ComputerVisionTools\datasets\mustache.PNG")
        dst = cv2.imread(r"C:\Users\hodda\PycharmProjects\Bootcamp\ComputerVisionTools\datasets\lips.PNG")

        for (x, y, w, h) in faces:
            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            L = y + h//2
            R = x + w//10
            img[L:dst.shape[0] + L, R:dst.shape[1] + R] = dst

            # cv2.imread("\datasets\mustache.PNG")
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            # eyes = eye_cascade.detectMultiScale(roi_gray)
            # for (ex, ey, ew, eh) in eyes:
            #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    dst = cv2.imread(r"C:\Users\hodda\PycharmProjects\Bootcamp\ComputerVisionTools\datasets\mustache.PNG", 0)
    img2 = cv2.imread(r'C:\Users\hodda\Downloads\messi_1.PNG', 0)

    # img2[] = dst

    # dst = cv2.addWeighted(img2, 0.7, img1, 0.3, 0)
    # img2[0:dst.shape[0], 0:dst.shape[1]] = dst
    # cv2.imshow('dst', img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    adding_drawing_on_face()


    print()