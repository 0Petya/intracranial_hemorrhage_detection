# This class will allow keras to load arrays in batches, as there are too many to fit in memory all at once.

import numpy as np

from tensorflow.keras.backend import expand_dims
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, ids, y, path, batch_size):
        self.ids = ids
        self.y = y
        self.path = path
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))

    def __getitem__(self, i):
        ids = self.ids[i * self.batch_size: (i + 1) * self.batch_size]
        y = self.y[i * self.batch_size:(i + 1) * self.batch_size]
        X = expand_dims(np.array([np.load(f"{self.path}/{id}.npy") for id in ids]).astype(np.float32), 3)
        return X, y
