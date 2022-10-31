import matplotlib.animation
from PIL import Image
import io
from data import GRID
import tqdm.notebook
from IPython.display import HTML
from base64 import b64encode
import numpy as np
import matplotlib.pyplot as plt


def show_decision_surface(model, X, y, ax=None):
    """
    Helper function to visualize the decision surface of the trained model
    :param model with predict method
    :return: None
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_grid, y_grid = np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1)
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


def create_buffer(X, assignments):
    plt.ioff()
    figure = plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=assignments)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    return buf


def show_progress(snapshots):
    num_snapshots = len(snapshots)
    figure = plt.figure()
    figure.set_figheight(5), figure.set_figwidth(8)

    def animate(t):
        t = int(t)
        plt.cla()
        snapshots[t].seek(0)
        im = Image.open(snapshots[t])
        plt.axis('off')
        plt.imshow(im)

    anim = matplotlib.animation.FuncAnimation(figure, animate, frames=num_snapshots)
    return anim


def save_frames(frames, target_mp4):
    import imageio
    imageio.mimwrite('{}.mp4'.format(target_mp4), np.array(frames, dtype="uint8"), fps=30)


def evaluate_policy(policy, target_mp4, random=False, n_episodes=10):
    env = GRID(random=False, max_time=100)
    frames = []
    for _ in tqdm.notebook.tqdm(range(n_episodes)):
        state = env.reset()
        game_over = False
        while not game_over:
            action = policy.act(state)
            state, _, game_over = env.step(policy.act(action))
        frames += env.episode
    save_frames(frames, "multi_episodes")


def display_video(target_mp4):
    mp4 = open('{}.mp4'.format(target_mp4), 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML("""
        <video width=320 controls>
            <source src="%s" type="video/mp4">
        </video>
        """ % data_url)
