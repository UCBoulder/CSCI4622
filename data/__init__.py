import os
import pickle

current_folder = os.path.dirname(os.path.abspath(__file__))


class HousePrices(object):
    def __init__(self):
        self.X, self.y = pickle.load(open(os.path.join(current_folder, 'prices.pkl'), 'rb'))
