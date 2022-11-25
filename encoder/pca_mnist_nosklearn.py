import numpy as np
from tensorflow.linalg import svd
from tensorflow import matmul
from tensorflow.keras.datasets import mnist

def pca(x, N):
    x = np.reshape(x, (x.shape[0], 784)).astype("float32") / 255.
    mean = x.mean(axis=0)
    x -= mean[None, :]

    s, u, v = svd(x)

    projM = v[:, 0:N]
    return mean, projM

def apply_pca(mean, projM, x):
    x = np.reshape(x, (x.shape[0], 784)).astype("float32") / 255.
    x -= mean[None, :]

    return matmul(x, projM)

(train_x, train_y), (test_x, test_y) = mnist.load_data()

N = 32

mean, projM = pca(train_x, N)

train_x_pca = apply_pca(mean, projM, train_x)
test_x_pca = apply_pca(mean, projM, test_x)
