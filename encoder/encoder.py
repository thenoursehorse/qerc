
import h5py
import numpy as np

from pathlib import Path
from timeit import default_timer as timer

import qutip as qt

class Encoder(object):
    def __init__(self, N, filename, save=True, H_dim=None, H_shape=None):
        self.N = N
        self.filename = filename
        self.save = save

        if H_dim == None:
            self.H_dim = 2 # qubit
        else:
            self.H_dim = H_dim
        
        if H_shape == None:
            self.H_shape = 2**self.N # qubit
        else:
            self.H_shape = H_shape

        self._psi0_train = None
        self._psi0_test = None
        self.filename_psi0 = None

    def to_Qobj(self, states):
        dims = [self.H_dim, [1 for _ in range(len(self.H_dim))] ]
        shape = (self.H_shape, 1)
        Nt = len(states)
        N_samples = states[0].shape[-1]
        return [[qt.Qobj(states[t][:,k], dims=dims, shape=shape) for k in range(N_samples)] for t in range(Nt)]

    def get_psi0(self, N_samples, test=False):
        # This is hacky but we are just using Qobj for its storage capabilities
        # and not actually using the proper tensor structure it additionally encodes
        #psi0 = qt.Qobj(shape=(self._H.shape[0], N_samples_train))

        # This shape so that can multiply U into it with a dangling N_samples index
        psi0 = np.empty(shape=(self.H_shape, N_samples), dtype=complex)
        
        print("Constructing all initial psi:")
        start_total = timer()
        start = timer()
        for k in range(N_samples):
            psi0[:,k,None] = self.encode_psi(k=k, test=test).full()
            if ((k%(1000)) == 0) and (k != 0):
                end = timer()
                print(f"sample {k}/{N_samples} took:", end-start)
                start = timer()
        end = timer()
        print(f"sample {k}/{N_samples} took:", end-start)
        #psi0 = qt.Qobj(psi0)
        end_total = timer()
        print("Duration:", end_total-start_total)
        print()

        return psi0

    def save_psi0(self):
        assert self.filename_psi0 != None, "There is no filename for psi0 set !"

        with h5py.File(self.filename_psi0, 'w') as f:
            f.create_dataset('psi0_train', data=self._psi0_train)
            f.create_dataset('psi0_test', data=self._psi0_test)
    
    def load_psi0(self):
        assert self.filename_psi0 != None, "There is no filename for psi0 set !"
        
        with h5py.File(self.filename_psi0, 'r') as f:
            self._psi0_train = np.array( f['psi0_train'] )
            self._psi0_test = np.array( f['psi0_test'] )
        
    @property
    def psi0_train(self):
        return self._psi0_train
    
    @property
    def psi0_test(self):
        return self._psi0_test