import numpy as np
import scipy
        
# From Fast Computation of Moore-Penrose Inverse Matrices by Pierre Courrieu 2005
# Returns the Moore-Penrose inverse of the argument
def _geninv(G):
    # Transpose if m < n
    m, n = G.shape
    transpose = False
    if m < n:
        transpose = true
        A = G @ G.conj().T
    else:
        A = G.conj().T @ G

    # Full rank Cholesky factorization of A
    L = np.linalg.cholesky(A)
    #L = scipy.linalg.cholesky(A, lower=True)
    
    # Computation of the generalized inverse of G
    M = np.linalg.inv(L.conj().T @ L)
    if transpose:
        Y = G.conj().T @ L @ M @ M @ L.conj().T
    else:
        Y = L @ M @ M @ L.conj().T @ G.conj().T
    return Y

# Regularized psuedo inverse for regularized ELM
def _reginv(A, C=1e10):
    m, n = A.shape
    if m <= n :
        left = A.conj().T @ A + (1.0/C)*np.eye(A.shape[-1])
        left_inv = np.linalg.inv(left)
        return left_inv @ A.conj().T
    else:
        right = A @ A.conj().T + (1.0/C)*np.eye(A.shape[0])
        right_inv = np.linalg.inv(right)
        return A.conj().T @ right_inv

class ELM(object):
    def __init__(self, input_size, hidden_size, activation='softmax', random='uniform', pinv='geninv'):
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._activation = activation
        self._random = random
        self._pinv = pinv

        rng = np.random.default_rng()

        if self._activation != 'identity':
            if self._random == 'uniform':
                self._rand = rng.uniform
                self._weight = self._rand(-1, 1, (self._hidden_size, self._input_size))
                self._bias = self._rand(-1, 1, (self._hidden_size))
           
            elif self._random == 'normal':
                self._rand = rng.standard_normal
                self._weight = self._rand((self._hidden_size, self._input_size))
                self._bias = self._rand((1, self._hidden_size))
            
            else:
                raise ValueError(f'unknown random function type {self._random}.')

        # Identity has no hidden layer, just optimizes the output weight beta
        else:
            self._rand = None
            self._weight = None
            self._bias = None

        self._H = 0
        self._beta = 0

    def activation_function(self, x):
        if self._activation == 'softmax':
            u = x @ self._weight.T + self._bias
            upper = np.exp(u)
            lower = np.sum(upper, axis=-1)
            out = np.empty(shape=upper.shape)
            for i in range(out.shape[0]):
                out[i,:] = upper[i,:] / lower[i]
            return out
        
        elif self._activation == 'sigmoid':
            u = x @ self._weight.T + self._bias
            return 1.0 / (1.0 + np.exp(-1.0 * u))
        
        elif self._activation == 'hyperbolic':
            u = x @ self._weight.T + self._bias
            a = np.exp(-u)
            return (1.0 - a) / (1.0 + a)

        elif self._activation == 'hard':
            u = x @ self._weight.T + self._bias
            idx_gr = u > 0
            idx_ls = u <= 0
            u[ idx_gr ] = 0
            u[ idx_ls ] = 1
            return u

        elif self._activation == 'cos':
            u = x @ self._weight.T + self._bias
            return np.cos(u)

        elif self._activation == 'identity':
            return x

        else:
            raise ValueError(f'unknown activation function {self.activation}.')

    def train(self, x, y):

        self._H = self.activation_function(x)

        if self._pinv == 'numpy': # SVD
            self._H_plus = np.linalg.pinv(self._H)

        elif self._pinv == 'scipy': # linear regression
            self._H_plus = scipy.linalg.pinv(self._H)
        
        elif self._pinv == 'jax':
            from jax.numpy.linalg import pinv as jpinv
            self._H_plus = jpinv(self._H)
        
        elif self._pinv == 'reginv':
            self._H_plus = _reginv(self._H)
        
        elif self._pinv == 'geninv':
            self._H_plus = _geninv(self._H)

        else:
            raise ValueError(f'unknown psuedo inverse function {self._pinv}')
    
        self._beta = self._H_plus @ y
        return self._H @ self._beta

    def evaluate(self, y_pred, y):
        # Majority vote for 1 hot vectors
        y_pred_trunc = np.argmax(y_pred, axis=-1)
        y_trunc = np.argmax(y, axis=-1)

        accuracy = np.sum(y_pred_trunc == y_trunc) / y.shape[0]
        mse = np.mean((y_pred - y)**2)
        mae = np.mean(np.abs(y_pred - y))
        
        return accuracy, mse, mae
    
    def predict(self, x):
        return self.activation_function(x) @ self.beta
    
    def onehot_y(self, y_train, y_test, M=10, binarizer=True):      
        # Use sklearn binarizer to make one hot vectors
        if binarizer:
            from sklearn.preprocessing import LabelBinarizer
            lb = LabelBinarizer()
            lb.fit(y_train)
            y_train = lb.transform(y_train).astype(int)
            y_test = lb.transform(y_test).astype(int)
            return y_train, y_test
        
        # Make one hot vectors of M categories
        else:
            N_samples_train = len(y_train)
            T_train = np.zeros(shape=(N_samples_train, M))
            x = [i for i in range(N_samples_train)]
            T_train[x,y_train] = 1
            
            N_samples_test = len(y_test)
            T_test = np.zeros(shape=(N_samples_test, M))
            x = [i for i in range(N_samples_test)]
            T_test[x,y_test] = 1
            
            return T_train, T_test
        
    def standardize(self, x_train, x_test):
        from sklearn.preprocessing import StandardScaler
        std_scalar = StandardScaler()
        
        # Standardize output of training set
        std_scalar.fit(x_train)
        x_train_std = std_scalar.transform(x_train)
        
        # Apply to test
        x_test_std = std_scalar.transform(x_test)
        
        return x_train_std, x_test_std
            
    @property
    def input_size(self):
        return self._input_size

    @property
    def hidden_size(self):
        return self._hidden_size
    
    @property
    def activation(self):
        return self._activation
    
    @property
    def random(self):
        return self._random
    
    @property
    def pinv(self):
        return self._pinv
    
    @property
    def weight(self):
        return self._weight
    
    @property
    def bias(self):
        return self._bias

    @property
    def H(self):
        return self.H

    @property
    def H_plus(self):
        return self.H_plus
    
    @property
    def beta(self):
        return self._beta
