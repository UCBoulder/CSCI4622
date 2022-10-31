from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from .grid import GRID
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

current_folder = os.path.dirname(os.path.abspath(__file__))


class HousePrices(object):
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = pickle.load(
            open(os.path.join(current_folder, 'test_train.pkl'), 'rb'))
        np.random.seed(10)
        scalers = np.random.randint(2, 100, (1, self.X_train.shape[1])) / 10.0
        b = np.random.normal(4, 10)
        self.X_test = self.X_test * scalers + b
        self.X_train = self.X_train * scalers + b


class BinaryDigits:
    """
    Class to store MNIST data for images of 9 and 8 only
    """

    def __init__(self):
        loaded = np.load(os.path.join(current_folder, "mnist.npz"))
        images = loaded["images"].reshape(-1, 28 * 28)
        labels = loaded["labels"]
        labels = labels % 2
        train_size = 1000
        valid_size = 500
        test_size = 500

        self.X_train, self.y_train = images[:train_size], labels[:train_size]
        self.X_valid, self.y_valid = images[train_size: train_size + valid_size], labels[
                                                                                  train_size: train_size + valid_size]
        self.X_test, self.y_test = (images[train_size + valid_size:train_size + valid_size + test_size],
                                    labels[train_size + valid_size: train_size + valid_size + test_size])
class Circles(object):
    def __init__(self):
        self.X, self.labels = make_circles(n_samples=400, noise=0.1, random_state=5622, factor=0.8)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.labels,
                                                                                test_size=0.3, random_state=5622)


class DataBlobs:
    def __init__(self, centers, std=1.75):
        self.X, self.labels = make_blobs(n_samples=400, n_features=2, cluster_std=std, centers=centers,
                                         shuffle=False, random_state=5622)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.labels,
                                                                                test_size=0.3, random_state=5622)


class DataMoons(object):
    def __init__(self):
        self.X, self.labels = make_moons(n_samples=400, noise=0.05, shuffle=False, random_state=5622)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.labels,
                                                                                test_size=0.3, random_state=5622)


def get_inception_layer(f1, f2, f3):
    """
    Returns an "naive" inception block. Works better when we want to detect pixel level details
    :param f1: num filters for 1x1 convolution
    :param f2: num filters for 2x2 convolution
    :param f3: num filters for 4x4 convolution
    :return:
    """
    from keras import layers
    class InceptionLayer(layers.Layer):
        def __init__(self, f1, f2, f3):
            super(InceptionLayer, self).__init__()

            self.conv1 = layers.Conv2D(f1, (1, 1), padding='same', activation='relu')
            self.conv2 = layers.Conv2D(f2, (2, 2), padding='same', activation='relu')
            self.conv4 = layers.Conv2D(f3, (4, 4), padding='same', activation='relu')
            self.pool = layers.MaxPooling2D((2, 2), strides=(1, 1), padding='same')
            self.concat = layers.Concatenate(axis=-1)

        def call(self, inputs, *args, **kwargs):
            return self.concat([self.conv1(inputs),
                                self.conv2(inputs),
                                self.conv4(inputs),
                                self.pool(inputs)])

    return InceptionLayer(f1, f2, f3)
