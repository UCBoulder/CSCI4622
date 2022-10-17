from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


class Circles(object):
    def __init__(self):
        self.X, self.labels = make_circles(n_samples=300, noise=0.1, random_state=5622, factor=0.6)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.labels,
                                                                                test_size=0.3, random_state=5622)


class DataBlobs:
    def __init__(self, centers, std=1.75):
        self.X, self.labels = make_blobs(n_samples=300, n_features=2, cluster_std=std, centers=centers,
                                         shuffle=False, random_state=5622)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.labels,
                                                                                test_size=0.3, random_state=5622)


class DataMoons(object):
    def __init__(self):
        self.X, self.labels = make_moons(n_samples=300, noise=0.05, shuffle=False, random_state=5622)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.labels,
                                                                                test_size=0.3, random_state=5622)
