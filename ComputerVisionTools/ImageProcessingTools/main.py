import cv2
import numpy as np


def viewImage(image, name_of_window):
    # cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(name_of_window)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def detection():
    img = cv2.imread(r"C:\Users\hodda\Downloads\fig_exp.PNG")
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # COLOR_MIN = np.array([20, 80, 80], np.uint8)
    # COLOR_MAX = np.array([40, 255, 255], np.uint8)
    COLOR_MIN = np.array([10, 20, 20], np.uint8)
    COLOR_MAX = np.array([50, 255, 255], np.uint8)
    frame_threshed = cv2.inRange(hsv_img, COLOR_MIN, COLOR_MAX)
    viewImage(hsv_img, "hsv")
    imgray = frame_threshed

    ret, thresh = cv2.threshold(frame_threshed, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find the index of the largest contour
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]

    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Show", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    img = cv2.imread(r"C:\Users\hodda\Downloads\plate.PNG")

