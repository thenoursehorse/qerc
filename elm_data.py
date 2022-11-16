import sys
import h5py
from timeit import default_timer as timer
from pathlib import Path 

import numpy as np
import scipy
        
#from numpy.linalg import pinv as pinv
from scipy.linalg import pinv as pinv
#from jax.numpy.linalg import pinv as pinv

#from numpy.linalg import inv as inv
from scipy.linalg import inv as inv
#from jax.numpy.linalg import inv as inv

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import accuracy_score

from mnist_qubit import PCAQubits
from observer import Observer


# From Fast Computation of Moore-Penrose Inverse Matrices by Pierre Courrieu 2005
# Returns the Moore-Penrose inverse of the argument
def geninv(G):
    # Transpose if m < n
    m, n = G.shape
    transpose = False
    if m < n:
        transpose = true
        A = G @ G.conj().T
    else:
        A = G.conj().T @ G

    # Full rank Cholesky factorization of A
    #L = np.linalg.cholesky(A)
    L = scipy.linalg.cholesky(A, lower=True)
    #_, L, _ = scipy.linalg.lu(A)
    
    # Computation of the generalized inverse of G
    M = inv(L.conj().T @ L)
    if transpose:
        Y = G.conj().T @ L @ M @ M @ L.conj().T
    else:
        Y = L @ M @ M @ L.conj().T @ G.conj().T
    return Y

# Regularized psuedo inverse for regularized ELM
def reginv(A, C=1e10):
    C = 1e10
    left = A.conj().T @ A + (1.0/C)*np.eye(A.shape[-1])
    left_inv = inv(left)
    return left_inv @ A.conj().T

class ELM(object):
    def __init__(self, input_size, output_size, hidden_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        rng = np.random.default_rng()
        #self.rand = rng.uniform
        self.rand = rng.standard_normal

        # FIXME using uniform can cause H @ H.T to not be positive-definite for large hidden_size
        # Initialize random weight with range [-0.5, 0.5]
        self.weight = self.rand(-0.5, 0.5, (self.hidden_size, self.input_size))
        #self.weight = self.rand(0, 1, (self.hidden_size, self.input_size))
        self.bias = self.rand(0, 1, (1, self.hidden_size))

        # Initialize random bias with range [0, 1]
        #self.bias = self.rand((1, self.hidden_size))
        #self.weight = self.rand((self.hidden_size, self.input_size))

        self.H = 0
        self.beta = 0

    # Activation function
    def activation_func(self, x):
        #a = np.exp(x)
        #return a / np.sum(a)                        # softmax
        
        return 1.0 / (1.0 + np.exp(-1.0 * x))      # sigmoid
        
        #a = np.exp(-x)
        #return (1.0 - a) / (1.0 + a)               # hyperbolic tangent

    def predict(self, x):
        x = np.array(x)
        return self.activation_func((x @ self.weight.T) + self.bias) @ self.beta

    def train(self, x, y):

        x = np.array(x)
        y = np.array(y)

        self.H = self.activation_func((x @ self.weight.T) + self.bias)

        # Classical ELM
        #self.H_plus = inv(self.H.T @ self.H) @ self.H.T # bad if poorly conditioned
        #self.H_plus = pinv(self.H)
        #self.H_plus = reginv(self.H)
        self.H_plus = geninv(self.H)
    
        self.beta = self.H_plus @ y
        return self.H @ self.beta

class ELMData(object):
    def __init__(self, N, g):
        self.N = N
        self.g = g
        
        self.input_size = 2**N
        self.output_size = None

        self.lb = None
        self.std_scalar = None

        self.y_train = None
        self.y_test = None
        self.x_train = None
        self.x_test = None
        self.x_train_std = None
        self.x_test_std = None
    
        self.directory_train = f'data/ising_train_N_{self.N}_g_{self.g:0.3f}'
        self.directory_test = f'data/ising_test_N_{self.N}_g_{self.g:0.3f}'
            
        self.filename_train = self.directory_train+"/"+f'ising_train_N_{self.N}_g_{self.g:0.3f}'
        self.filename_test = self.directory_test+"/"+f'ising_test_N_{self.N}_g_{self.g:0.3f}'
            
        self.filename_train_x = self.filename_train+"_x.h5"
        self.filename_test_x = self.filename_test+"_x.h5"

    def get_y(self, N_samples, test=False):
        pca = PCAQubits(N=self.N)

        if test:
            self.y_test = pca.test_y
        else:
            self.y_train = pca.train_y
        
        # Just use the integer to fit
        #return y[:N_samples,None].astype(int), None
        
        # Make one hot vectors
        #M = 10 # categories
        #T = np.zeros(shape=(N_samples, M))
        #x = [i for i in range(N_samples)]
        #T[x,y[:N_samples]] = 1
        #return T, None
      
        if not test:
            # Use sklearn binarizer to make one hot vectors
            self.lb = LabelBinarizer()
            self.lb.fit(self.y_train)
            self.y_train = self.lb.transform(self.y_train)[:N_samples,:].astype(int)
            self.output_size = self.y_train.shape[1]
            return self.y_train
        else:
            self.y_test = self.lb.transform(self.y_test)[:N_samples,:].astype(int)
            return self.y_test
        
    def get_x(self, N_samples, test=False):
        if test:
            filename = self.filename_test
        else:
            filename = self.filename_train

        # The time evolved samples
        observer = Observer(N=self.N, filename=filename, load=True)
        
        # The input nodes
        x = np.empty(shape=(N_samples, self.input_size))
        
        start = timer()
        for k in range(N_samples):
            
            observer.load_h5(k=k)
        
            n = -1 # For now just take the end time
            #zz = observer.zz[:,n]
            #z = observer.z[:,n]
            #x[k,:] = [*zz, *z]

            rho_diag = observer.rho_diag[:,n]
            x[k,:] = [*rho_diag]
           
            if ((k%(1000)) == 0) and (k != 0):
                end = timer()
                print(f"g={g} sample {k}/{N_samples} took:", end-start)
                start = timer()
        end = timer()
        print(f"g={g} sample {k}/{N_samples} took:", end-start)
        
        # FIXME it isn't clear to me that standardizing the output makes it 
        # work better
        if not test:
            self.x_train = x
            
            # Standardize output of training set
            self.standardize(self.x_train)
            self.x_train_std = self.std_scalar.transform(self.x_train)
            return self.x_train
        else:
            self.x_test = x
            
            self.standardize(self.x_train) # because must fit off train data
            self.x_test_std = self.std_scalar.transform(self.x_test)
            return self.x_test

    def standardize(self, x):
        self.std_scalar = StandardScaler()
        self.std_scalar.fit(x)

    def save_x(self, test=False):        
        if test:
            filename = self.filename_test_x
        else:
            filename = self.filename_train_x
    
        with h5py.File(filename, 'w') as f:
            if test:
                f.create_dataset('x', data=self.x_test)
                f.create_dataset('x_std', data=self.x_test_std)
            else:
                f.create_dataset('x', data=self.x_train)
                f.create_dataset('x_std', data=self.x_train_std)

    def load_x(self, test=False):
        if test:
            filename = self.filename_test_x
        else:
            filename = self.filename_train_x

        with h5py.File(filename, 'r') as f:
            if test:
                self.x_test = np.array( f['x'] )
                self.x_test_std = np.array( f['x_std'] )
            else:
                self.x_train = np.array( f['x'] )
                self.x_train_std = np.array( f['x_std'] )
            
if __name__ == '__main__':
    # The parameters
    print("Usage: python3 optimize_elm.py <N> <g> <N_samples_train> <N_samples_test> <hidden_size>")
    if len(sys.argv) != 6:
        sys.exit(1)
    print('Argument List:', str(sys.argv))
    N = int(sys.argv[1])
    g = float(sys.argv[2])
    N_samples_train = int(sys.argv[3])
    N_samples_test = int(sys.argv[4])
    hidden_size = int(sys.argv[5])

    if N_samples_train < 0:
        N_samples_train = 60000 # number of training samples
    
    if N_samples_test < 0:
        N_samples_test = 10000 # number of training samples
     
    # Output data
    elm_data = ELMData(N=N, g=g)
    y_train = elm_data.get_y(N_samples=N_samples_train)
    y_test = elm_data.get_y(N_samples=N_samples_test, test=True)
    
    # Input data
    print("filename_train_x: ", elm_data.filename_train_x)
    x_file = Path(elm_data.filename_train_x)
    if not x_file.is_file():
        print("x_train data does not exist")
        x_train = elm_data.get_x(N_samples=N_samples_train)
        elm_data.save_x()
    else:
        print("x_train data exists")
        elm_data.load_x()
        x_train = elm_data.x_train
    
    print("filename_test_x: ", elm_data.filename_test_x)
    x_file = Path(elm_data.filename_test_x)
    if not x_file.is_file():
        print("x_test data does not exist")
        x_test = elm_data.get_x(N_samples=N_samples_test, test=True)
        elm_data.save_x(test=True)
    else:
        print("x_test data exists")
        elm_data.load_x(test=True)
        x_test = elm_data.x_test

    input_size = elm_data.input_size
    output_size = elm_data.output_size
    
    # Number of hidden neurons in the elm part
    if hidden_size == -1:
        hidden_size = 784
    elif hidden_size < -1:
        hidden_size = input_size

    elm = ELM(input_size, output_size, hidden_size)
    
    # Train the ELM
    start = timer()
    y = elm.train(x_train, y_train)
    end = timer()
    print(f"g={g} elm took:", end-start)
    print("norm error: ",  np.linalg.norm(y - y_train))
    
    y_train_trunc = (y > 0.5).astype(int)  # for 1 hot vectors
    #y_trunc = np.round(y).astype(int) # for integers
    print("Training accuracy: ", accuracy_score(y_train, y_train_trunc))

    y_test_trunc = (elm.predict(x_test) > 0.5).astype(int)  # for 1 hot vectors
    print("Testing accuracy: ", accuracy_score(y_test, y_test_trunc))

