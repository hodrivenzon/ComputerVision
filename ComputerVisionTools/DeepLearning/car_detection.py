import numpy as np
from PIL import ImageGrab
import tensorflow as tf
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4
import cv2
import time

from datetime import datetime
from time import gmtime, strftime, localtime
import pandas as pd




def load_image_to_tensor(file):
    # load image
    image = tf.io.read_file(file)
    # detect format (JPEG, PNG, BMP, or GIF) and converts to Tensor:
    image = tf.io.decode_image(image)
    return image

def screen_record():
    last_time = time.time()
    while(True):
        # 800x600 windowed mode
        printscreen = np.array(ImageGrab.grab(bbox=(600, 400, 1400, 900)))
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        cv2.imshow('window',cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        return printscreen

def detected_photo(boxes, scores, classes, detections,image):
    HEIGHT, WIDTH = (image.shape[0]//32)*32, (image.shape[1]//32)*32
    boxes = (boxes[0] * [WIDTH, HEIGHT, WIDTH, HEIGHT]).astype(int)
    scores = scores[0]
    classes = classes[0].astype(int)
    detections = detections[0]

    CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
        'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    ########################################################################

    image_cv = image.numpy()

    # for (xmin, ymin, xmax, ymax), score, class_idx in zip(boxes, scores, classes):
    #
    #     if score > 0:
    #         # if class_idx == 2:         # show bounding box only to the "car" class
    #
    #         #### Draw a rectangle ##################
    #         # convert from tf.Tensor to numpy
    #         cv2.rectangle(image_cv, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,255,0), thickness= 2)
    #         # Add detection text to the prediction
    #         text = CLASSES[class_idx] + ': {0:.2f}'.format(score)
    #         cv2.putText(image_cv, text, (int(xmin), int(ymin) - 5),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    #
    # return image_cv
    cars_count = 0
    for (xmin, ymin, xmax, ymax), score, class_idx in zip(boxes, scores, classes):

        if score > 0:
            if class_idx == 2 or class_idx == 5 or class_idx == 7 or class_idx == 3 or class_idx == 1:         # show bounding box only to the "car" class

                #### Draw a rectangle ##################
                # convert from tf.Tensor to numpy
                cv2.rectangle(image_cv, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,255), thickness= 2)
                # Add detection text to the prediction
                text = CLASSES[class_idx] + ': {0:.2f}'.format(score)
                cv2.putText(image_cv, text, (int(xmin), int(ymin) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                cars_count += 1
    print(f"cars_count:{cars_count}")
    return image_cv, cars_count

def resize_image(image):
    # image = np.array(ImageGrab.grab(bbox=(600, 400, 1400, 900)))
    HEIGHT, WIDTH = (image.shape[0]//32)*32, (image.shape[1]//32)*32
    # Resize the output_image:
    image = tf.image.resize(image, (HEIGHT, WIDTH))
    # Add a batch dim:
    images = tf.expand_dims(image, axis=0)/255
    return images

def get_image_from_plot():
    # crates a numpy array from the output_image of the plot\figure
    canvas = FigureCanvasAgg(Figure())
    canvas.draw()
    return np.fromstring(canvas.tostring_rgb(), dtype='uint8')

def trained_yolov4_model():
    # load trained yolov4 model
    image = np.array(ImageGrab.grab(bbox=(600, 400, 1400, 900)))
    # image = np.array(ImageGrab.grab(bbox=(600, 400, 1400, 900)))
    HEIGHT, WIDTH = (image.shape[0]//32)*32, (image.shape[1]//32)*32
    model = YOLOv4(
        input_shape=(HEIGHT, WIDTH, 3),
        anchors=YOLOV4_ANCHORS,
        num_classes=80,
        # training=False,
        yolo_max_boxes=20,
        yolo_iou_threshold=0.5,
        yolo_score_threshold=0.73,
    )
    model.load_weights('yolov4.h5')
    return model

def proccess_frame(photo, model):
    images = resize_image(photo)
    boxes, scores, classes, detections = model.predict(images)
    result_img, cars_count = detected_photo(boxes, scores, classes, detections, images[0])
    return result_img, cars_count

def main():
    yolo_model = trained_yolov4_model()
    last_time = time.time()
    cars_count_list = []
    cars_count_old = 0
    video_stopped_count = 0
    repeat = 0

    while(True):
        now = datetime.now()
        # 800x600 windowed mode
        printscreen = np.array(ImageGrab.grab(bbox=(600, 400, 1400, 900)))
        # print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        cv2.imshow('window', cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))

        image, cars_count = proccess_frame(printscreen, yolo_model)
        if video_stopped_count < 2:
            cars_count_list.append([now.strftime("%d_%m_%Y_%H_%M_%S"), cars_count])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow('window', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if cars_count_old == cars_count:
            if cars_count != 0:
                repeat += 1
                if repeat > 5:
                    video_stopped_count += 1
                    repeat = 0
            else:
                video_stopped_count += 0.1
        else:
            video_stopped_count = 0
            repeat = 0
        cars_count_old = cars_count
        print("\n")
        print(f"video_stopped_count: {video_stopped_count}")
        print(f"repeat: {repeat}")
        print(f"sec_passed: {len(cars_count_list)}")
        # print(str(now.strftime("%d_%m_%Y_%H_%M_%S")))

        patience = 50
        file_size = 3600

        if len(cars_count_list) > file_size:
            pd.DataFrame(cars_count_list, columns=['date', 'cars_count'])\
                .to_csv(r'cars_count_files/cars_count_'
                        + str(now.strftime("%d_%m_%Y_%H_%M_%S")) +'.csv', index=False)
            print("[INFO] File saved to CSV.")
            print("[INFO] Cause: Video length larger than 1 hour.")
            cars_count_list = []

        backup_file_size = (file_size // 100)

        if len(cars_count_list) % backup_file_size == 0:
            pd.DataFrame(cars_count_list, columns=['date', 'cars_count'])\
                .to_csv(r'cars_count_files/backup_files/cars_count_backup_' + str(len(cars_count_list) // backup_file_size) + '.csv', index=False)
            print("[INFO] File saved to backup CSV.")

        if (cv2.waitKey(25) & 0xFF == ord('q')) or video_stopped_count > patience:
            pd.DataFrame(cars_count_list, columns=['date', 'cars_count'])\
                .to_csv(r'cars_count_files/cars_count_'
                        + str(now.strftime("%d_%m_%Y_%H_%M_%S")) +'.csv', index=False)
            print("[INFO] File saved to CSV.")
            if video_stopped_count > patience:
                print("[INFO] Cause: Video stopped.")
            else:
                print("[INFO] Cause: Stopped by User.")
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()