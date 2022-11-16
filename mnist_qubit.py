import h5py
import numpy as np

from pathlib import Path

import qutip as qt

class PCAQubits:
    def __init__(self, N, filename='data/mnist_pca', load=True):
        self.N = N
        self.filename = filename + f'_N_{N}.h5'

        if load == True:
            if Path(self.filename).is_file():
                self.load()
            else:
                self.get_pca()
                self.save()

    def get_pca(self, N=None):
        from keras.datasets import mnist
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        if N == None:
            N = self.N

        (self.train_x, self.train_y), (self.test_x, self.test_y) = mnist.load_data()
        self.train_x = self.train_x.reshape(-1, 784).astype("float32")# / 255.
        self.test_x = self.test_x.reshape(-1, 784).astype("float32")# / 255.

        # Standardize pixels and transform
        std_scalar = StandardScaler()
        std_scalar.fit(self.train_x)
        self.train_x_scalar = std_scalar.transform(self.train_x)
        self.test_x_scalar = std_scalar.transform(self.test_x)

        # keep 2*N coponents (N is # of qubits)
        pca = PCA(2*N)
        #var = .95 # keep N components so fit is 95% variance retained
        #pca = PCA(var)

        # Find pca representation
        pca.fit(self.train_x_scalar)
        self.train_pca = pca.transform(self.train_x_scalar)
        self.test_pca = pca.transform(self.test_x_scalar)

        # Select the appropriate components for each
        self.train_theta = self.train_pca[:,:N]
        self.train_phi = self.train_pca[:,N:2*N]
        
        self.test_theta = self.test_pca[:,:N]
        self.test_phi = self.test_pca[:,N:2*N]

        # Normalize to the range 0-pi
        theta_min = np.min(self.train_theta)
        theta_max = np.max(self.train_theta)
        self.train_theta = np.pi * (self.train_theta - theta_min ) / (theta_max - theta_min)
        self.test_theta = np.pi * (self.test_theta - theta_min ) / (theta_max - theta_min)
        
        phi_min = np.min(self.train_phi)
        phi_max = np.max(self.train_phi)
        self.train_phi = np.pi * (self.train_phi - phi_min ) / (phi_max - phi_min)
        self.test_phi = np.pi * (self.test_phi - phi_min ) / (phi_max - phi_min)

        # Truncate test data to make sure it fits in 0-pi range
        idx_max = np.where(self.test_theta > np.pi)
        idx_min = np.where(self.test_theta < 0)
        self.test_theta[idx_max] = np.pi
        self.test_theta[idx_min] = 0
        
        idx_max = np.where(self.test_phi > np.pi)
        idx_min = np.where(self.test_phi < 0)
        self.test_phi[idx_max] = np.pi
        self.test_phi[idx_min] = 0

    def save(self, filename=None):
        if filename == None:
            filename = self.filename
        
        with h5py.File(filename, 'w') as f:
            f.create_dataset('train_theta', data=self.train_theta)
            f.create_dataset('train_phi', data=self.train_phi)
            f.create_dataset('train_y', data=self.train_y)
            
            f.create_dataset('test_theta', data=self.test_theta)
            f.create_dataset('test_phi', data=self.test_phi)
            f.create_dataset('test_y', data=self.test_y)

    def load(self, filename=None):
        if filename == None:
            filename = self.filename
        
        with h5py.File(filename, 'r') as f:
            self.train_theta = np.array( f['train_theta'] )
            self.train_phi = np.array( f['train_phi'] )
            self.train_y = np.array( f['train_y'] )
            
            self.test_theta = np.array( f['test_theta'] )
            self.test_phi = np.array( f['test_phi'] )
            self.test_y = np.array( f['test_y'] )

    def encode_qubit(self,theta, phi):
        spin_up = qt.basis(2,0)
        spin_dn = qt.basis(2,1)
        return np.cos(theta)*spin_up + np.exp(1j*phi)*np.sin(theta)*spin_dn

    def encode_psi(self, theta, phi):
        assert len(theta) == len(phi), f"len(theta) = len({theta}), len(phi) = len({phi}), have to be equal !"
        qubits = [self.encode_qubit(theta[i], phi[i]) for i in range(len(theta))]
        return qt.tensor(*qubits)
