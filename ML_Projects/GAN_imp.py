import numpy as np
from keras.datasets import mnist
from tqdm import tqdm
import matplotlib.pyplot as plt

from keras.layers import Input
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras import initializers

np.random.seed(1000)
randomDim = 100


# 1. Build input random input images using random.random
def noise_gen(batchSize, randomDim):
    return np.random.normal(0, 1, size=[batchSize, randomDim]);


# 2. Build a generator network that transforms these images to another image, the “fake image”.
# Feel free to build the architecture as you wish.
def create_generator():
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
    return generator

# 3. Build a discriminator network that accepts an image and determmins weather it is a MNIST image or a fake image
def create_discriminator():
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
    adam = Adam(lr=0.0002, beta_1=0.5)
    discriminator.compile(loss='binary_crossentropy', optimizer=adam)
    return discriminator

# Combined network
def create_gan():
    discriminator = create_discriminator()
    discriminator.trainable = False
    ganInput = Input(shape=(randomDim,))

    generator = create_generator()
    x = generator(ganInput)
    ganOutput = discriminator(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)

    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan, discriminator, generator


def get_data():
    # Load MNIST data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28 * 28)
    X_test = X_test.reshape(X_test.shape[0], 28 * 28)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    X_train /= 255
    X_test /= 255
    return X_train, y_train, X_test, y_test


def train(X_train, epochs=1, batchSize=128):
    dLosses = []
    gLosses = []
    batchCount = int(X_train.shape[0] / batchSize)
    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)
    # 4. Build a training loop in which you iterate between a training step on the Generator
    # network and a training step on the discriminator network
    gan, discriminator, generator = create_gan()
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
            # evaluation(generator)

        # Store loss of most recent batch from this epoch
        evaluation(generator, title=f"Epoch: {e}")
        dLosses.append(dloss)
        gLosses.append(gloss)

    return generator


def evaluation(generator, title=None):
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
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.pause(0.1)
    plt.close()

def main():
    X_train, y_train, X_test, y_test = get_data()
    generator = train(X_train, epochs=25, batchSize=128)
    evaluation(generator)

if __name__ == '__main__':
    main()




