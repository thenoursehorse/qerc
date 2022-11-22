import numpy as np
import scipy
        
# From Fast Computation of Moore-Penrose Inverse Matrices by Pierre Courrieu 2005
# Returns the Moore-Penrose inverse of the argument
# The C factor is because cholesky requires a positive definite matrix,
# and sometimes this is not satisfied on the order of 1e-12. Regularizing it
# as in _reginv produces minimal error
# This is fast because the cholesky decomposition is orders of magnitude faster 
# than a full svd for very large matrices
# Even for C = 0 this has a slight error compared to svd sometimes?
# I think when A is not positive defininite introducing any round off error
# has big affects on training accuracy (even if the psuedo inverse is numerically
# very close to what is obtained with svd)
def _geninv(G, C=1e-10):
    # Transpose if m < n
    m, n = G.shape
    transpose = False
    if m < n:
        transpose = true
        A = G @ G.conj().T + C*np.eye(G.shape[0])
    else:
        A = G.conj().T @ G + C*np.eye(G.shape[1])

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

# From http://dx.doi.org/10.1155/2014/641706
# Still not as fast as geninv
# But as accurate as svd (if don't use k), but just as slow
# Also does not seem to run into problems with not being positive defininite
def _IMqrginv(A, eps=None):
    m, n = A.shape
    Q, R = np.linalg.qr(A)
    if eps != None:
        eps = 1e-5 # recommendation from paper
        # The k below introduces error for not much greater speed up
        k = np.sum(np.any(np.abs(R) > epse, axis=1))
        Q = Q[:,:k]
        R = R[:k,:]
    M = np.linalg.inv( R @ R.conj().T )
    return R.conj().T @ M @ Q.conj().T
    #return np.linalg.inv(R) @ Q.conj().T

# Regularized psuedo inverse for regularized ELM
def _reginv(A, C=1e-10):
    m, n = A.shape
    if m <= n :
        left = A.conj().T @ A + C*np.eye(A.shape[-1])
        left_inv = np.linalg.inv(left)
        return left_inv @ A.conj().T
    else:
        right = A @ A.conj().T + C*np.eye(A.shape[0])
        right_inv = np.linalg.inv(right)
        return A.conj().T @ right_inv

# Iterative method It24C2 from https://doi.org/10.1016/j.cam.2010.08.042
# This isn't faster than svd or geninv, and not very accurate
def _iterinv(A, beta=1.0, eps=1e-10, maxiter=2000):
    alpha = 1.0 / (np.linalg.norm(A, ord=np.inf) * np.linalg.norm(A, ord=1))
    #alpha = beta

    # initialize Xk
    Xk = alpha*A.conj().T

    minnormd = np.inf
    normd1 = np.inf

    for i in range(maxiter):
        Xk1 = (1.0+beta) * Xk - beta*Xk @ A @ Xk

        normd = np.linalg.norm(Xk1 - Xk)
        #normd = np.linalg.norm( A @ Xk1 @ A - A )

        if normd < minnormd:
            minnormd = normd
            Xkmin = Xk

        #if np.abs(normd/normd1 - 1.0 - beta) < eps:
        if np.abs(normd) < eps:
            break

        normd1 = normd
        Xk = Xk1

    print(f"took {i} iterations")
    return Xkmin

# Iterative method "An iterative method to compute Moore-Penrose inverse based 
# on gradient maximal convergence rate" by Sheng and Wang 2013
# This is very unstable for poorly chosen beta. How is beta estimated in the 
# original paper? It is not clear to me. Regardless, it is much slower than
# other methods because of how many matrix products one has to take
def _iterinv2(A, beta=0.0001, eps=1e-10, maxiter=2000):
    alpha = 1.0

    # Initialize Xk
    Xk = alpha * A.conj().T
        
    minnormd = np.inf

    for i in range(maxiter):
        Xk1 = Xk + beta * A.conj().T @ (A - A @ Xk @ A) @ A.conj().T

        normd = np.linalg.norm(Xk1 - Xk)
        #normd = np.linalg.norm( A @ Xk1 @ A - A )
        
        if normd < minnormd:
            minnormd = normd
            Xkmin = Xk1

        if np.abs(normd) < eps:
            break

        Xk = Xk1

    print(f"took {i} iterations")
    return Xkmin

class ELM(object):
    def __init__(self, input_size, hidden_size, activation='softmax', random='uniform', pinv='geninv', C=1e-10):
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._activation = activation
        self._random = random
        self._pinv = pinv
        self._C = C # regularization factor for pinv methods

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
            self._H_plus = _reginv(self._H, C=self._C)

        elif self._pinv == 'IMqrginv':
            self._H_plus = _IMqrginv(self._H)

        elif self._pinv == 'geninv':
            self._H_plus = _geninv(self._H, C=self._C)

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
