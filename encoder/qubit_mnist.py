import h5py
import numpy as np

from pathlib import Path

import qutip as qt

from .encoder import Encoder

class PCAQubits(Encoder):
    def __init__(self, load=True, **kwargs):
        super().__init__(**kwargs)
        self.filename_pca = self.filename + f'qubit_pca_mnist_N_{self.N}.h5'
        self.filename_psi0 = self.filename + f'qubit_psi0_mnist_N_{self.N}.h5'

        if load == True:
            if Path(self.filename_pca).is_file():
                self.load_pca()
            else:
                self.get_pca()
                if self.save:
                    self.save_pca()
            
            if Path(self.filename_psi0).is_file():
                self.load_psi0()
            else:
                print("Training:")
                self._psi0_train = self.get_psi0(N_samples=self.theta_train.shape[0], test=False)
                print("Testing:")
                self._psi0_test = self.get_psi0(N_samples=self.theta_test.shape[0], test=True)
                if self.save:
                    self.save_psi0()

    def get_pca(self, N=None):
        #from keras.datasets import mnist
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        if N == None:
            N = self.N

        #(self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        #self.x_train = self.x_train.reshape(-1, 784).astype("float32") / 255.0
        #self.x_test = self.x_test.reshape(-1, 784).astype("float32") / 255.0

        mnist = fetch_openml('mnist_784', as_frame=False)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)
        self.y_train = self.y_train.astype(int)
        self.y_test = self.y_test.astype(int)

        # Standardize pixels and transform
        std_scalar = StandardScaler()
        std_scalar.fit(self.x_train)
        self.x_train_scalar = std_scalar.transform(self.x_train)
        self.x_test_scalar = std_scalar.transform(self.x_test)

        # keep 2*N coponents (N is # of qubits)
        pca = PCA(2*N)
        #pca = PCA(.95) # keep N components so fit is 95% variance retained

        # Find pca representation
        pca.fit(self.x_train_scalar)
        self.pca_train = pca.transform(self.x_train_scalar)
        self.pca_test = pca.transform(self.x_test_scalar)

        # Select the appropriate components for each
        self.theta_train = self.pca_train[:,:N]
        self.phi_train = self.pca_train[:,N:2*N]
        
        self.theta_test = self.pca_test[:,:N]
        self.phi_test = self.pca_test[:,N:2*N]

        # Normalize to the range 0-pi
        pca_min = np.min(self.pca_train)
        pca_max = np.max(self.pca_test)
        self.theta_train = np.pi * (self.theta_train - pca_min ) / (pca_max - pca_min)
        self.theta_test = np.pi * (self.theta_test - pca_min ) / (pca_max - pca_min)
        self.phi_train = np.pi * (self.phi_train - pca_min ) / (pca_max - pca_min)
        self.phi_test = np.pi * (self.phi_test - pca_min ) / (pca_max - pca_min)

        # Normalize to the range 0-pi
        #theta_min = np.min(self.theta_train)
        #theta_max = np.max(self.theta_train)
        #self.theta_train = np.pi * (self.theta_train - theta_min ) / (theta_max - theta_min)
        #self.theta_test = np.pi * (self.theta_test - theta_min ) / (theta_max - theta_min)
        
        #phi_min = np.min(self.phi_train)
        #phi_max = np.max(self.phi_train)
        #self.phi_train = np.pi * (self.phi_train - phi_min ) / (phi_max - phi_min)
        #self.phi_test = np.pi * (self.phi_test - phi_min ) / (phi_max - phi_min)

        # Truncate test data to make sure it fits in 0-pi range
        idx_max = np.where(self.theta_test > np.pi)
        idx_min = np.where(self.theta_test < 0)
        self.theta_test[idx_max] = np.pi
        self.theta_test[idx_min] = 0
        
        idx_max = np.where(self.phi_test > np.pi)
        idx_min = np.where(self.phi_test < 0)
        self.phi_test[idx_max] = np.pi
        self.phi_test[idx_min] = 0

    def save_pca(self):
        with h5py.File(self.filename_pca, 'w') as f:
            f.create_dataset('theta_train', data=self.theta_train)
            f.create_dataset('phi_train', data=self.phi_train)
            f.create_dataset('y_train', data=self.y_train)
            
            f.create_dataset('theta_test', data=self.theta_test)
            f.create_dataset('phi_test', data=self.phi_test)
            f.create_dataset('y_test', data=self.y_test)

    def load_pca(self):
        with h5py.File(self.filename_pca, 'r') as f:
            self.theta_train = np.array( f['theta_train'] )
            self.phi_train = np.array( f['phi_train'] )
            self.y_train = np.array( f['y_train'] )
            
            self.theta_test = np.array( f['theta_test'] )
            self.phi_test = np.array( f['phi_test'] )
            self.y_test = np.array( f['y_test'] )

    def encode_qubit(self,theta, phi):
        spin_up = qt.basis(2,0)
        spin_dn = qt.basis(2,1)
        return np.cos(0.5*theta)*spin_up + np.exp(1j*phi)*np.sin(0.5*theta)*spin_dn

    def encode_psi(self, k, test=False):
        if test:
            theta = self.theta_test[k]
            phi = self.phi_test[k]
        else:
            theta = self.theta_train[k]
            phi = self.phi_train[k]
        qubits = [self.encode_qubit(theta[i], phi[i]) for i in range(len(theta))]
        return qt.tensor(*qubits)