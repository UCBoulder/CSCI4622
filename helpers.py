import matplotlib.pylab as plt
import numpy as np


def plot_iris(X, y, ax=None):
    # Plot the dataset
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    name_color_dict = {
        0: ("steelblue", "setosa"),
        1: ("#a76c6e", "versicolor"),
        2: ("#6a9373", "virginica")
    }
    for k in [0, 1, 2]:
        ax.scatter(X[y == k, 0], X[y == k, 1], color=name_color_dict[k][0],
                   s=50, label=name_color_dict[k][1])
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=16)
    ax.set_xlabel("sepal length (cm)", fontsize=16)
    ax.set_ylabel("sepal width (cm)", fontsize=16)
    return ax


def plot_decision_surface(X, model, ax):
    # Plot the decision boundary.
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.025), np.arange(y_min, y_max, 0.025))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Define custom colormap
    # cmap = colors.ListedColormap(['steelblue', '#a76c6e', '#6a9373'])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.5, zorder=1)
