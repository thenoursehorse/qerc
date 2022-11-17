import h5py
from timeit import default_timer as timer

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer

from ../mnist_qubit import PCAQubits
from ../observer import Observer

# FIXME make just a filename that you pass
class ELMData(object):
    def __init__(self, N, g, N_samples_train=None, N_samples_test=None):
        self.N_samples_train = N_samples_train
        self.N_samples_test = N_samples_test

        self.N = N
        self.g = g


        
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

    def get_y(self, N_samples_train=None, N_samples_test=None):
        if N_samples_train == None:
            N_samples_train = self.N_samples_train
        if N_samples_test == None:
            N_samples_test = self.N_samples_test
        
        pca = PCAQubits(N=self.N)
        self.y_train = pca.train_y
        self.y_test = pca.test_y
        
        # Just use the integer to fit
        #return y[:N_samples,None].astype(int), None
        
        # Make one hot vectors
        #M = 10 # categories
        #T = np.zeros(shape=(N_samples, M))
        #x = [i for i in range(N_samples)]
        #T[x,y[:N_samples]] = 1
        #return T, None
      
        # Use sklearn binarizer to make one hot vectors
        self.lb = LabelBinarizer()
        self.lb.fit(self.y_train)
        self.y_train = self.lb.transform(self.y_train)[:N_samples_train,:].astype(int)
            
        self.output_size = self.y_train.shape[1]
            
        self.y_test = self.lb.transform(self.y_test)[:N_samples_test,:].astype(int)
            
        return self.y_train, self.y_test

    def get_x_observer(self, N, filename, N_samples):
        # The time evolved samples
        observer = Observer(N=N, filename=filename, load=True)
        
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
        return x
        
        
    def get_x(self, N_samples_train=False, N_samples_test=False):
        if N_samples_train == None:
            N_samples_train = self.N_samples_train
        if N_samples_test == None:
            N_samples_test = self.N_samples_test
        
        if test:
            filename = self.filename_test
        else:
            filename = self.filename_train
    
        self.x_train = get_x_observer(self.N, filename, N_samples):

        # FIXME it isn't clear to me that standardizing the output makes it 
        # work better
        self.x_train = x
            
        # Standardize output of training set
        self.standardize(self.x_train)
        self.x_train_std = self.std_scalar.transform(self.x_train)
        return self.x_train
        self.x_test = x
            
        self.standardize(self.x_train) # because must fit off train data
        self.x_test_std = self.std_scalar.transform(self.x_test)
        return self.x_test

    def standardize(self, x):
        self.std_scalar = StandardScaler()
        self.std_scalar.fit(x)

    def save_x(self):        
        with h5py.File(filename, 'w') as f:
            f.create_dataset('x_test', data=self.x_test)
            f.create_dataset('x_test_std', data=self.x_test_std)
            f.create_dataset('x_train', data=self.x_train)
            f.create_dataset('x_train_std', data=self.x_train_std)

    def load_x(self, test=False):
        with h5py.File(filename, 'r') as f:
            self.x_test = np.array( f['x_test'] )
            self.x_test_std = np.array( f['x_test_std'] )
            self.x_train = np.array( f['x_train'] )
            self.x_train_std = np.array( f['x_train_std'] )            