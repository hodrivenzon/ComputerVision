import os, cv2, keras  # cv2 =to perform selective search on the images,  run= pip install opencv-contrib-python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras import Model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ModelCheckpoint, EarlyStopping

path = r"datasets\Images"
annot = r"datasets\Airplanes_Annotations\Airplanes_Annotations"


def selective_search_on_images(path, annot):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    train_images = []
    train_labels = []  # label= airplane images =1 and non airplane images (i.e. background images) = 0

    for e, i in enumerate(os.listdir(annot)):
        try:
            #            if e>20:
            #                break
            if i.startswith("airplane"):
                filename = i.split(".")[0] + ".jpg"
                #                print(e,filename)
                image = cv2.imread(os.path.join(path, filename))
                df = pd.read_csv(os.path.join(annot, i))
                gtvalues = []
                for row in df.iterrows():
                    x1 = int(row[1][0].split(" ")[0])
                    y1 = int(row[1][0].split(" ")[1])
                    x2 = int(row[1][0].split(" ")[2])
                    y2 = int(row[1][0].split(" ")[3])
                    gtvalues.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
                ss.setBaseImage(
                    image)  # Loop over the image folder and set each image one by one as the base for selective search
                ss.switchToSelectiveSearchFast()  # Initialising fast selective search and getting proposed regions
                ssresults = ss.process()
                imout = image.copy()
                counter = 0
                falsecounter = 0
                flag = 0
                fflag = 0
                bflag = 0
                for e, result in enumerate(
                        ssresults):  # Iterating over all the first 2000 results passed by selective search
                    if e < 2000 and flag == 0:
                        for gtval in gtvalues:
                            x, y, w, h = result  # calculating IOU of the proposed region and annotated region
                            iou = get_iou(gtval, {"x1": x, "x2": x + w, "y1": y, "y2": y + h})
                            if counter < 30:  # we will collect maximum of 30 negative sample (i.e. background)
                                if iou > 0.70:  # and positive sample (i.e. airplane) from one image
                                    timage = imout[y:y + h, x:x + w]
                                    resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                                    train_images.append(resized)
                                    train_labels.append(1)  # label= airplane images =1
                                    counter += 1
                            else:
                                fflag = 1
                            if falsecounter < 30:  # we will collect maximum of 30 negative sample (i.e. background)
                                if iou < 0.3:  # and positive sample (i.e. airplane) from one image
                                    timage = imout[y:y + h, x:x + w]
                                    resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                                    cv2.imshow("img", resized)
                                    cv2.waitKey(1000)
                                    cv2.destroyAllWindows()
                                    train_images.append(resized)
                                    train_labels.append(0)  # label= non airplane images (i.e. background images) = 0
                                    falsecounter += 1
                            else:
                                bflag = 1
                        if fflag == 1 and bflag == 1:
                            print("inside")
                            flag = 1
        except Exception as e:
            print(e)
            print("error in " + filename)
            continue

    x_new = np.array(train_images)
    y_new = np.array(train_labels)  # label= airplane images =1 and non airplane images (i.e. background images) = 0
    return x_new, y_new


def get_iou(bb1, bb2):
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    assert iou >= 0.0
    assert iou <= 1.0
    return iou


##################################################################################################################
# Transfer Learning:
# instead of training all the layers of the model we lock some of the layers and use those trained weights in the
# locked layers to extract particular features from our data

def create_model_for_transfer_learning():
    vggmodel = VGG16(weights='imagenet', include_top=True)  # put the imagenet weight in the model
    # vggmodel.summary()         # weights for our whole model will be downloaded. If this is set to false then
    # the pre-trained weights will only be downloaded for convolution layers and
    # no weights will be downloaded for dense layers.

    for layers in (vggmodel.layers)[
                  :18]:  # no update weights, take only 15 first layers, will not be training the weights of the first 15 layers
        # print(layers)                      # and use it as it is
        layers.trainable = False
        # last dense layer of my model should be a 2 unit softmax dense layer. Take the second last layer of the model, adding a dense softmax layer of 2 units in the end
    X = vggmodel.layers[
        -2].output  # take the second last layer, remove the last layer of the VGG16 model which is made to predict 1000 classes
    predictions = Dense(2, activation="softmax")(
        X)  # Dense(2)= two classes so the last dense layer of model should be a 2 unit softmax dense layer
    model_final = Model(input=vggmodel.input, output=predictions)
    opt = Adam(lr=0.0001)
    model_final.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=["accuracy"])
    # model_final.summary()   # we have a 2 unit dense layer in the end so i will be using categorical_crossentropy as loss
    return model_final


class MyLabelBinarizer(LabelBinarizer):  # for one-hot encoding of the label

    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1 - Y))
        else:
            return Y


def predict_on_one_image(model_loaded, image_num):
    # To do prediction on that model

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    z = 0
    for e, i in enumerate(os.listdir(path)):
        if i.startswith(str(image_num)):
            z += 1
            img = cv2.imread(os.path.join(path, i))  # pass the image from selective search
            ss.setBaseImage(img)
            ss.switchToSelectiveSearchFast()
            ssresults = ss.process()
            imout = img.copy()
            for e, result in enumerate(ssresults):
                if e < 2000:
                    x, y, w, h = result
                    timage = imout[y:y + h, x:x + w]
                    resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                    img = np.expand_dims(resized, axis=0)
                    out = model_loaded.predict(img)  # pass all the result of the selective search to the model as input
                    if out[0][
                        0] > 0.70:  # If the output of the model says the region to be a foreground image (i.e. airplane image)
                        # proposed region in which the accuracy of the model was above 0.70.
                        cv2.rectangle(imout, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
                        # do localisation on an image and perform object detection
                        # create bounding box on the original image on the coordinate of the proposed region
    return imout


if __name__ == '__main__':
    x_new, y_new = selective_search_on_images(path, annot)
    lenc = MyLabelBinarizer()
    Y = lenc.fit_transform(y_new)  # # for one-hot encode the label

    x_train, x_test, y_train, y_test = train_test_split(x_new, Y, test_size=0.10)

    # to pass the dataset to the model
    trdata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=0)
    traindata = trdata.flow(x=x_train, y=y_train)
    tsdata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=0)
    testdata = tsdata.flow(x=x_test, y=y_test)

    checkpoint = ModelCheckpoint("ieeercnn_vgg16_1.h5", monitor='val_loss', verbose=1,
                                 save_best_only=True, save_weights_only=False, mode='auto', period=1)

    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')

    model_final = create_model_for_transfer_learning()
    # start the training of the model
    hist = model_final.fit_generator(generator=traindata,
                                     steps_per_epoch=10,
                                     epochs=5,
                                     validation_data=testdata,
                                     validation_steps=2,
                                     callbacks=[early])

    imout = predict_on_one_image(model_final, 4)
    plt.figure()
    plt.imshow(imout)





