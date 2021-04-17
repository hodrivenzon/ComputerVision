import cv2
import numpy as np
from scipy.spatial import distance

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
def sharpening():
    # generating the kernels
    kernel_sharpen_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # kernel_sharpen_2 = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])
    # kernel_sharpen_3 = np.array([[-1, -1, -1, -1, -1],
    #                              [-1, 2, 2, 2, -1],
    #                              [-1, 2, 8, 2, -1],
    #                              [-1, 2, 2, 2, -1],
    #                              [-1, -1, -1, -1, -1]]) / 8.0

    # output_1 = cv2.filter2D(img, -1, kernel_sharpen_1)
    # output_2 = cv2.filter2D(img, -1, kernel_sharpen_2)
    # output_3 = cv2.filter2D(img, -1, kernel_sharpen_3)
    # cv2.imshow('Sharpening', output_1)
    # cv2.imshow('Excessive Sharpening', output_2)
    # cv2.imshow('Edge Enhancement', output_3)
    # cv2.waitKey(0)

if __name__ == '__main__':

    img = cv2.imread(r"C:\Users\hodda\Downloads\plate.PNG")
    # img = cv2.imread(r"C:\Users\hodda\Downloads\plate_2.PNG")
    img = cv2.imread(r"C:\Users\hodda\Downloads\plate_3.PNG")
    # img = cv2.imread(r"C:\Users\hodda\Downloads\plate_4.PNG")
    # img = cv2.imread(r"C:\Users\hodda\Downloads\plate_5.PNG")
    # viewImage(img, "Image")


    scale_percent = 75  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # viewImage(resized, "resized")

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, threshold_image = cv2.threshold(resized, 127, 255, 0)
    # viewImage(gray_image, "gray_image")
    blurred = cv2.blur(img, (7, 7))
    viewImage(blurred, "Blurred")
    blurred = cv2.GaussianBlur(blurred, (5, 5), 9)


    viewImage(blurred, "Gauss Blurred")


    edges = cv2.Canny(blurred, 50, 200)
    viewImage(edges, "edges")
    h = edges.shape[0]
    w = edges.shape[1]
    x_edge = edges.mean(axis=0) - edges.mean(axis=0).min()
    y_edge = edges.mean(axis=1) - edges.mean(axis=1).min()

    for i, point in enumerate(x_edge):
        if point != 0 and i > 10:
            print(f"x_s:{i}")
            x_s = i
            break
    for i, point in enumerate(np.flip(x_edge)):
        if point != 0 and i > 10:
            print(f"x_e:{i}")
            x_e = i
            break
    for i, point in enumerate(y_edge):
        if point != 0 and i > 10:
            print(f"y_s:{i}")
            y_s = i
            break
    for i, point in enumerate(np.flip(y_edge)):
        if point != 0 and i > 10:
            print(f"y_e:{i}")
            y_e = i
            break

    p_1 = (x_s, (h-y_e+y_s)//2)
    p_2 = (w-x_e, (h - y_e + y_s) // 2)
    p_3 = ((w - x_e + x_s) // 2, y_s)
    p_4 = ((w - x_e + x_s) // 2, h-y_e)
    print(p_1)
    print(p_2)
    print(p_3)
    print(p_4)

    # np.sqrt(np.sum((np.subtract(p_1, p_2))**2))
    # p_center = tuple(np.int64(
    #     np.add(np.sqrt(np.subtract(p_1, p_2) ** 2) // 2, np.sqrt(np.subtract(p_3, p_4) ** 2) // 2)))
    # p_center = tuple(np.int64(np.add(np.sqrt(np.subtract(p_1, p_2) ** 2), np.sqrt(np.subtract(p_3, p_4) ** 2))))
    p_center = ((w - x_e + x_s) // 2, (h - y_e + y_s) // 2)
    # p_center = tuple(np.int64(np.add(np.add(np.sqrt(np.subtract(p_1, p_2) ** 2)//2, p_1), np.sqrt(np.subtract(p_3, p_4) ** 2)//2)))
    print(p_center)
    green = (0, 255, 0)
    # color = (255, 255, 255)
    color = (100, 100, 100)
    color = green
    # cv2.imshow("Show", gray_image)
    cv2.line(gray_image, p_1, p_2, green)
    cv2.line(gray_image, p_3, p_4, green)
    # cv2.circle(gray_image, p_1, radius=5, color=color)
    # cv2.circle(gray_image, p_2, radius=5, color=color)
    # cv2.circle(gray_image, p_3, radius=5, color=color)
    # cv2.circle(gray_image, p_4, radius=5, color=color)
    cv2.circle(gray_image, p_center, radius=5, color=color)

    # cv2.circle(edges, (h // 2, w // 2), radius=5, color=(255, 255, 255))
    # cv2.circle(gray_image, (w // 2, h // 2), radius=5, color=color)
    # cv2.imshow("Show", edges)
    cv2.imshow("Show", gray_image)

    cv2.waitKey()
    cv2.destroyAllWindows()
    print()
