import numpy as np
import os
from sklearn.model_selection import train_test_split as _split

current_folder = os.path.dirname(os.path.abspath(__file__))


class Dataset:
    X_train = None
    y_train = None
    X_valid = None
    y_valid = None
    X_test = None
    y_test = None

    def __init__(self, data, labels, test_ratio=0.3, valid_ratio=0.0, random_seed=4622):
        if test_ratio > 0.0:
            self.X_train, self.X_test, self.y_train, self.y_test = _split(data, labels,
                                                                          test_size=test_ratio,
                                                                          random_state=random_seed)
        else:
            self.X_train, self.y_train = data, labels
        if valid_ratio > 0.0:
            self.X_train, self.X_valid, self.y_train, self.y_valid = _split(self.X_train, self.y_train,
                                                                            test_size=valid_ratio,
                                                                            random_state=random_seed)


class BinaryData(Dataset):

    def __init__(self, sqrt_n_samples: int = 24, random_state: int = 42):
        np.random.seed(random_state)

        self._sqrt_n_samples = sqrt_n_samples
        self._n_samples = sqrt_n_samples ** 2
        x, y = np.meshgrid(np.linspace(-1, 1, sqrt_n_samples), np.linspace(-1, 2, sqrt_n_samples))
        data = np.array([x, y]).reshape(2, -1).T

        data = data + 0.05 * np.random.uniform(-1, 1, (self._n_samples, 2))
        boundary = np.cos(1.5 * data[:, 0]) - 0.5
        self.x = data[:, 0]
        labels = 1 * (boundary < data[:, 1])

        flip = np.random.choice([0, 1], size=self._n_samples, p=[0.7, 0.3])
        # close_to_boundary = np.logical_and(0 < boundary - data[:, 1], boundary - data[:, 1] < 0.3)
        close_to_boundary = np.abs(boundary - data[:, 1]) < 0.3
        flip = np.logical_and(flip, close_to_boundary)
        labels[flip] = 1 - labels[flip]

        super(BinaryData, self).__init__(data, labels, test_ratio=0.3, valid_ratio=0.4)
        self.y_train = np.random.choice([0, 1], p=[0.1, 0.9]) * self.y_train

    def boundary(self):
        x = np.sort(self.x)
        return x, np.cos(1.5 * x) - 0.5


class DigitData(Dataset):

    def __init__(self):
        loaded = np.load(os.path.join(current_folder, "mnist.npz"))
        super(DigitData, self).__init__(loaded["images"], loaded["labels"], test_ratio=0.25, valid_ratio=0.3)


class SeparableData(Dataset):

    def __init__(self, num_samples=100, margin=0.1, random_seed=42):
        np.random.seed(random_seed)
        labels = np.random.choice([-1, 1], num_samples)
        data = np.zeros((num_samples, 2))
        num_pos = np.sum(labels == 1)
        num_neg = num_samples - num_pos
        flip = np.random.choice([-1, 1])
        data[labels==1] = np.random.uniform([-1 / 2 ** .5, margin + flip * 0.1], [1 / 2 ** .5, 1 / 2 ** .5],
                                      (num_pos, 2))
        data[labels==1][-1,-1] = margin + flip * 0.1
        data[labels==-1] = np.random.uniform([-1 / 2 ** .5, -1 / 2 ** .5], [1 / 2 ** .5, -margin + flip * 0.1],
                                      (num_neg, 2))

        data[labels==-1][-1, -1] = -margin + flip * 0.1
        data = np.dot(data, np.array([[np.cos(np.pi / 6), np.sin(np.pi / 6)], [-np.sin(np.pi / 6), np.cos(np.pi / 6)]]))

        super(SeparableData, self).__init__(data, labels, test_ratio=0.0, )
