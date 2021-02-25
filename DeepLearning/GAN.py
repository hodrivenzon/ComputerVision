import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras.layers import Input
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras import initializers

np.random.seed(1000)
randomDim = 100

class MyCallBack(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.08:
            print("\nepoch end: Loss is low so cancelling training!")
            self.model.stop_training = True

    def on_batch_end(self, batch, logs={}):
        if logs.get('loss') < 0.05:
            print("\nbatch end: Loss is low so cancelling training!")
            self.model.stop_training = True

class GAN:

    def __init__(self):
        pass

    def noise_gen(batchSize, randomDim):
        return np.random.normal(0, 1, size=[batchSize, randomDim])

    def main(self):
        generator = Sequential()
        generator.add(Dense(256, input_dim=randomDim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(512))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(1024))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(784, activation='tanh'))
        # Optimizer
        adam = Adam(lr=0.0002, beta_1=0.5)
        generator.compile(loss='binary_crossentropy', optimizer=adam)

        # 3. Build a discriminator network that accepts an image and determmins weather it is a MNIST image or a fake image
        discriminator = Sequential()
        discriminator.add(Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))
        discriminator.add(Dense(512))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))
        discriminator.add(Dense(256))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))
        discriminator.add(Dense(1, activation='sigmoid'))
        discriminator.compile(loss='binary_crossentropy', optimizer=adam)

        # Combined network
        discriminator.trainable = False
        ganInput = Input(shape=(randomDim,))
        x = generator(ganInput)
        ganOutput = discriminator(x)
        gan = Model(inputs=ganInput, outputs=ganOutput)
        gan.compile(loss='binary_crossentropy', optimizer=adam)

        dLosses = []
        gLosses = []

        # Load MNIST data
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0], 28 * 28)
        X_test = X_test.reshape(X_test.shape[0], 28 * 28)
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        X_train /= 255
        X_test /= 255

        epochs = 1
        batchSize = 128

        batchCount = int(X_train.shape[0] / batchSize)
        print('Epochs:', epochs)
        print('Batch size:', batchSize)
        print('Batches per epoch:', batchCount)
        # 4. Build a training loop in which you iterate between a training step on the Generator
        # network and a training step on the discriminator network

        for e in range(1, epochs + 1):
            print('-' * 15, 'Epoch %d' % e, '-' * 15)
            for _ in tqdm(range(batchCount)):
                # Get a random set of input noise and images
                noise = np.random.normal(0, 1, size=[batchSize, randomDim])
                imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchSize)]

                # Generate fake MNIST images
                generatedImages = generator.predict(noise)
                # print np.shape(imageBatch), np.shape(generatedImages)
                X = np.concatenate([imageBatch, generatedImages])

                # Labels for generated and real data
                yDis = np.zeros(2 * batchSize)
                # One-sided label smoothing
                yDis[:batchSize] = 0.9

                # Train discriminator
                discriminator.trainable = True
                dloss = discriminator.train_on_batch(X, yDis)

                # Train generator
                noise = np.random.normal(0, 1, size=[batchSize, randomDim])
                yGen = np.ones(batchSize)
                discriminator.trainable = False
                gloss = gan.train_on_batch(noise, yGen)

            # Store loss of most recent batch from this epoch
            dLosses.append(dloss)
            gLosses.append(gloss)

        examples = 5
        dim = (1, 5)
        figsize = (10, 10)
        noise = np.random.normal(0, 1, size=[examples, randomDim])
        generatedImages = generator.predict(noise)
        generatedImages = generatedImages.reshape(examples, 28, 28)

        plt.figure(figsize=figsize)
        for i in range(generatedImages.shape[0]):
            plt.subplot(dim[0], dim[1], i + 1)
            plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
            plt.axis('off')
        plt.tight_layout()
        plt.show()



    # def train(self, epochs=1, batchSize=128):
    #     batchCount = int(X_train.shape[0] / batchSize)
    #     print('Epochs:', epochs)
    #     print('Batch size:', batchSize)
    #     print('Batches per epoch:', batchCount)
    #     # 4. Build a training loop in which you iterate between a training step on the Generator
    #     # network and a training step on the discriminator network
    #     for e in range(1, epochs + 1):
    #         print('-' * 15, 'Epoch %d' % e, '-' * 15)
    #         for _ in tqdm(range(batchCount)):
    #             # Get a random set of input noise and images
    #             noise = np.random.normal(0, 1, size=[batchSize, randomDim])
    #             imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchSize)]
    #
    #             # Generate fake MNIST images
    #             generatedImages = generator.predict(noise)
    #             # print np.shape(imageBatch), np.shape(generatedImages)
    #             X = np.concatenate([imageBatch, generatedImages])
    #
    #             # Labels for generated and real data
    #             yDis = np.zeros(2 * batchSize)
    #             # One-sided label smoothing
    #             yDis[:batchSize] = 0.9
    #
    #             # Train discriminator
    #             discriminator.trainable = True
    #             dloss = discriminator.train_on_batch(X, yDis)
    #
    #             # Train generator
    #             noise = np.random.normal(0, 1, size=[batchSize, randomDim])
    #             yGen = np.ones(batchSize)
    #             discriminator.trainable = False
    #             gloss = gan.train_on_batch(noise, yGen)
    #
    #         # Store loss of most recent batch from this epoch
    #         dLosses.append(dloss)
    #         gLosses.append(gloss)




    def CNN(self):
        callbacks = MyCallBack()

        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        train_images = train_images.reshape(60000, 28, 28, 1)
        train_images = train_images / 255.0
        test_images = test_images.reshape(10000, 28, 28, 1)
        test_images = test_images / 255.0

        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        # model.fit(train_images, train_labels, epochs=5, callbacks=[callbacks])
        model.fit(train_images, train_labels, epochs=1)

        score = model.evaluate(test_images, test_labels)

        print(f'Score:{score}')

        model.save('epic_num_reader.model')

        new_model = keras.models.load_model('epic_num_reader.model')

        predictions = new_model.predict(test_images)

        for i in range(5):
            print(f'predicted digit:{np.argmax(predictions[i])}')
            print()
            plt.imshow(test_images[i].reshape(28, 28), cmap=plt.cm.binary)
            plt.show()