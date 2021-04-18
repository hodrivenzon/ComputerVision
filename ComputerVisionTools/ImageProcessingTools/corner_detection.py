import cv2
import numpy as np
from scipy.spatial import distance

def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':

    img = cv2.imread(r"C:\Users\hodda\Downloads\plate.PNG")
    # img = cv2.imread(r"C:\Users\hodda\Downloads\plate_2.PNG")
    # img = cv2.imread(r"C:\Users\hodda\Downloads\plate_3.PNG")
    img = cv2.imread(r"C:\Users\hodda\Downloads\plate_4.PNG")
    # img = cv2.imread(r"C:\Users\hodda\Downloads\plate_5.PNG")
    # img = cv2.imread(r"C:\Users\hodda\Downloads\car.PNG")
    # img = cv2.imread(r"C:\Users\hodda\Downloads\model.PNG")
    # img = cv2.imread(r"C:\Users\hodda\Downloads\elephant.PNG")

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners = cv2.goodFeaturesToTrack(image=gray, maxCorners=100, qualityLevel=0.1, minDistance=10)
    corners = np.int0(corners)
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)

    cv2.imshow('Corner', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

