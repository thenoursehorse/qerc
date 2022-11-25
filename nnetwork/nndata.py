import h5py
import numpy as np

class NNData(object):
    def __init__(self, filename, **kwargs):
        self.filename = filename
        for key in kwargs.keys():
            self.__dict__[key] = kwargs[key]
    
    def save(self):
        with h5py.File(self.filename, 'w') as f:
            for key in self.__dict__.keys():
                if key != 'filename':
                    f.create_dataset(key, data=self.__dict__[key])

    def load(self):
        with h5py.File(self.filename, 'r') as f:
            for key in f.keys():
                self.__dict__[key] = np.array(f[key])