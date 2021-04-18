import cv2
import numpy as np
from scipy.spatial import distance

def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_object_edges(img):

    blurred = cv2.blur(img, (5, 5))
    blurred = cv2.GaussianBlur(blurred, (7, 7), 9)
    edges = cv2.Canny(blurred, 50, 200)
    return edges

def find_object_margins(edges):
    h = edges.shape[0]
    w = edges.shape[1]
    x_edge = edges.mean(axis=0) - edges.mean(axis=0).min()
    y_edge = edges.mean(axis=1) - edges.mean(axis=1).min()

    try:
        for i, point in enumerate(x_edge):
            if point != 0 and i > 10:
                # print(f"x_s:{i}")
                x_s = i
                break
        for i, point in enumerate(np.flip(x_edge)):
            if point != 0 and i > 10:
                # print(f"x_e:{i}")
                x_e = i
                break
        for i, point in enumerate(y_edge):
            if point != 0 and i > 10:
                # print(f"y_s:{i}")
                y_s = i
                break
        for i, point in enumerate(np.flip(y_edge)):
            if point != 0 and i > 10:
                # print(f"y_e:{i}")
                y_e = i
                break

        p_1 = (x_s, (h-y_e+y_s)//2)
        p_2 = (w-x_e, (h - y_e + y_s) // 2)
        p_3 = ((w - x_e + x_s) // 2, y_s)
        p_4 = ((w - x_e + x_s) // 2, h-y_e)
        p_center = ((w - x_e + x_s) // 2, (h - y_e + y_s) // 2)

        return p_1, p_2, p_3, p_4, p_center
    except:
        print("[Error] Couldn't find object edges - try different blurring kernel, or other image.")



def draw_object_center(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = find_object_edges(img)
    # viewImage(edges, "edges")
    img_to_show = img
    p_1, p_2, p_3, p_4, p_center = find_object_margins(edges)

    color = (0, 0, 255)

    cv2.line(img_to_show, p_1, p_2, color)
    cv2.line(img_to_show, p_3, p_4, color)
    cv2.circle(img_to_show, p_1, radius=5, color=color)
    cv2.circle(img_to_show, p_2, radius=5, color=color)
    cv2.circle(img_to_show, p_3, radius=5, color=color)
    cv2.circle(img_to_show, p_4, radius=5, color=color)
    cv2.circle(img_to_show, p_center, radius=5, color=color)

    viewImage(img_to_show, "Show")

if __name__ == '__main__':

    img = cv2.imread(r"C:\Users\hodda\Downloads\plate.PNG")
    # img = cv2.imread(r"C:\Users\hodda\Downloads\plate_2.PNG")
    # img = cv2.imread(r"C:\Users\hodda\Downloads\plate_3.PNG")
    # img = cv2.imread(r"C:\Users\hodda\Downloads\plate_4.PNG")
    # img = cv2.imread(r"C:\Users\hodda\Downloads\plate_5.PNG")
    # img = cv2.imread(r"C:\Users\hodda\Downloads\car.PNG")
    img = cv2.imread(r"C:\Users\hodda\Downloads\model.PNG")
    # img = cv2.imread(r"C:\Users\hodda\Downloads\elephant.PNG")

    draw_object_center(img)

