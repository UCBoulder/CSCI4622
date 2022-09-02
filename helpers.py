import numpy as np
import matplotlib.pyplot as plt
import data


def show_decision_surface(model, X, y, ax=None):
    """
    Helper function to visualize the decision surface of the trained model
    Args:
        model:
        X: features matrix
        y: labels
        ax: subplot to plot on if provided
    """
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    x_grid, y_grid = np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05)
    xx, yy = np.meshgrid(x_grid, y_grid)
    r1, r2 = xx.reshape(-1, 1), yy.reshape(-1, 1)
    grid = np.hstack((r1, r2))
    y_hat = model.predict(grid).reshape(-1, )
    zz = y_hat.reshape(xx.shape)

    if ax is None:
        plt.contourf(xx, yy, zz, cmap='PiYG')
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.show()
    else:
        ax.contourf(xx, yy, zz, cmap='PiYG')
        ax.scatter(X[:, 0], X[:, 1], c=y)


def display_confusion(c_matrix):
    """
    Displays the confusion matrix using matrix show
    Args:
        c_matrix: square confusion matrix, shape (num_classes, num_classes)
    """
    _, ax = plt.subplots()
    ax.matshow(c_matrix, cmap=plt.cm.Blues)
    for i in range(c_matrix.shape[0]):
        for j in range(c_matrix.shape[0]):
            ax.text(i, j, str(c_matrix[j, i]), va='center', ha='center')
    plt.show()


def view_digit(digit_index, dataset: data.Dataset, partition, ax=None):
    """
   Display a digit given its index and partition
    Args:
        digit_index: index of the digit image
        dataset:
        partition:partition from which the digit is retrieved, either "train", "valid" or "test"
        ax:
    """
    image = {"train": dataset.X_train, "valid": dataset.X_valid, "test": dataset.X_test}[partition][digit_index]
    label = {"train": dataset.y_train, "valid": dataset.y_valid, "test": dataset.y_test}[partition][digit_index]
    image = image.reshape(28, 28)
    if ax is None:
        plt.figure()
        plt.matshow(image)
        plt.title("Digit %i" % label)
        plt.show()
    else:
        ax.matshow(image)
        ax.set_title("Digit %i" % label)


def plot_data(X, y):
    """
    Plots the simulated data.  Plots the learned decision boundary (#TODO)
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    colors = ["steelblue" if yi == -1 else "#a76c6e" for yi in y]
    ax.scatter(X[:, 0], X[:, 1], color=colors, s=75)
    ax.grid(alpha=0.25)
    ax.set_xlabel(r"$x_1$", fontsize=16)
    ax.set_ylabel(r"$x_2$", fontsize=16)
    plt.show()
