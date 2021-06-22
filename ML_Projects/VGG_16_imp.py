import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

import matplotlib.pyplot as plt
import os


def VGG_16():
    model = Sequential()
    model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=2, activation="softmax"))

    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
    model.summary()
    return model


def VGG16_transfer_learning():
    vggmodel = VGG16(weights='imagenet', include_top=True)
    for layers in vggmodel.layers[:19]:  # lock first 19 layers
        # print(layers)
        layers.trainable = False
    X = vggmodel.layers[-2].output
    predictions = Dense(2, activation="softmax")(X)
    model_final = Model(inputs=vggmodel.input, outputs=predictions)
    model_final.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.0001, momentum=0.9),
                        metrics=["accuracy"])
    return model_final


# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = os.path.basename(__file__).split('.')[0]
    plt.savefig(filename + '_plot.png')
    plt.close()


if __name__ == '__main__':

    folder_path = r'datasets/cats_and_dogs_filtered'

    target_size = (224, 224)
    epochs = 1
    batch_size = 16

    use_keras = True
    complete_training = False

    # create data generator
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # prepare iterators
    train_it = datagen.flow_from_directory(folder_path + r'\train' + '\\',
                                           class_mode='categorical', batch_size=batch_size, target_size=target_size)
    test_it = datagen.flow_from_directory(folder_path + r'\validation' + '\\',
                                          class_mode='categorical', batch_size=batch_size, target_size=target_size)



    # define model
    if complete_training:
        model = VGG_16()
    else:
        model = VGG16_transfer_learning()

    if use_keras:

        # fit model
        checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='auto', save_freq='epoch')
        early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')
        # tb = TensorBoard(log_dir='logs', )
        history = model.fit(train_it, steps_per_epoch=len(train_it),
                            validation_data=test_it, validation_steps=1, epochs=epochs, callbacks=[checkpoint, early])

        print('fit is over!')
        model.save_weights('vgg16_1.h5')

        # evaluate model
        _, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
        print('> %.3f' % (acc * 100.0))

        # learning curves
        # summarize_diagnostics(history)

    else:
        optimizer = Adam()
        loss_fn = CategoricalCrossentropy(from_logits=True)

        for epoch in range(epochs):
            for step in range(len(train_it)):
                print(f"Step: {step}/{len(train_it)}")
                with tf.GradientTape() as tape:
                    y_pred = model(train_it[step][0], training=True)
                    curr_loss = loss_fn(train_it[step][1], y_pred)
                grads = tape.gradient(curr_loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
            print(f'Epoch {epoch} loss {curr_loss.numpy()}')
