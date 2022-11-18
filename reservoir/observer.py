import numpy as np
import h5py
from timeit import default_timer as timer
from pathlib import Path

import qutip as qt

from .hamiltonians import get_spin_ops

def _vn_spectrum(rho, base=np.e, sparse=False):
    if rho.type == 'ket' or rho.type == 'bra':
        rho = qt.ket2dm(rho)
    vals = qt.sparse.sp_eigs(rho.data, rho.isherm, vecs=False, sparse=sparse)
    nzvals = vals[vals != 0]
    if base == 2:
        logvals = np.lib.scimath.log2(nzvals)
    elif base == np.e:
        logvals = np.lib.scimath.log(nzvals)
    else:
        raise ValueError("Base must be 2 or e.")
    return vals, logvals

def _entropy(N, psi, sparse=False):
    #half_ind = [0]
    half_ind = [i for i in range(int(N/2))]
    
    rhoA = psi.ptrace(half_ind, sparse=sparse)
    
    vals, logvals = _vn_spectrum(rhoA)

    #entropy_entanglement = qt.entropy.entropy_vn(rhoA, sparse=sparse)
    # mutual is 2*entropy_entanglement for pure states
    #rho = qt.states.ket2dm(psi)
    #half_compliment_ind = [half_ind[i]+int(N/2) for i in range(int(N/2))]
    #half_compliment_ind = [i+1 for i in range(N-1)]
    #entropy_mutual = qt.entropy.entropy_mutual(rho, half_ind, half_compliment_ind, sparse=sparse)
    return vals, float(np.real(-sum(vals[vals != 0] * logvals)))

class Observer:
    def __init__(self, N,
                       filename, 
                       N_samples_train=None, 
                       N_samples_test=None,
                       save=True,
                       load=False):
        self._N = N
        self._filename = filename
        
        self._N_samples_train = N_samples_train
        self._N_samples_test = N_samples_test
        
        self._save = save
        self._load = load

        self._result = None
        self._Nt = None

        if self._load:
            self.load_h5(filename=self._filename)

        self._corr_size = int( (self._N*self._N - self._N)/2 )
        if not self._load:
            self._x_ops = get_spin_ops(N=self._N, axis='x')
            self._z_ops = get_spin_ops(N=self._N, axis='z')
            self._xx_ops = [self._x_ops[i] * self._x_ops[j] for i in range(self._N) for j in range(self._N) if i > j]
            self._zz_ops = [self._z_ops[i] * self._z_ops[j] for i in range(self._N) for j in range(self._N) if i > j]
 
    def initialize(self, N_samples):
        self._x = np.empty(shape=(self._Nt, N_samples, self._N))
        self._z = np.empty(shape=(self._Nt, N_samples, self._N))
        self._xx = np.empty(shape=(self._Nt, N_samples, self._corr_size))
        self._zz = np.empty(shape=(self._Nt, N_samples, self._corr_size))
        self._ee = np.empty(shape=(self._Nt, N_samples))
        self._es = np.empty(shape=(self._Nt, N_samples, 2**int(self._N/2)))
        self._psi = np.empty(shape=(self._Nt, N_samples, 2**self._N), dtype=complex)

    def time_to_idx(self, time):
        return np.where( np.abs(self._tlist - time) < 1e-10)[0][0]
        
    def observe_one(self, psi_list, k):
        assert isinstance(psi_list, list), "psi must be a list of Qobj !"

        # qt.expect returns indexed as obs, time, hence transpose
        self._x[:,k,:] = np.array(qt.expect(oper=self._x_ops, state=psi_list)).T
        self._z[:,k,:] = np.array(qt.expect(oper=self._z_ops, state=psi_list)).T
        self._xx[:,k,:] = np.array(qt.expect(oper=self._xx_ops, state=psi_list)).T
        self._zz[:,k,:] = np.array(qt.expect(oper=self._zz_ops, state=psi_list)).T
            
        #zz = self.zz[:,n]
        #z = self.z[:,n]
        #x[k,:] = [*zz, *z]

        for n in range(len(psi_list)):
            psi = psi_list[n]
            self._es[n,k,:], self._ee[n,k] = _entropy(self._N, psi)
   
        for n in range(len(psi_list)):
            psi = psi_list[n]
            #self._rho_diag[n,k,:] = (psi.full().conj() * psi.full())[:,0].real
            #if psi.type == 'ket' or psi.type == 'bra':
            #    rho = qt.ket2dm(psi)
            #self._rho_diag[:,n] = np.real(rho.data.diagonal())
           
            # All coefficients in the computational basis
            self._psi[n,k,:] = psi.full()[:,0]

    def observe_all(self):
        N_samples_train = self._N_samples_train
        N_samples_test = self._N_samples_test
        
        print("Observing all training samples:")
        filename = self._filename+'_train'
        
        # Check time dimension of the first sample
        self._result = self.load_qu(filename=filename, k=0)
        self.initialize(N_samples=N_samples_train)

        # Take observations
        start = timer()
        for k in range(N_samples_train):
            self._result = self.load_qu(filename=filename, k=k)
            self.observe_one(psi_list=self._result.states, k=k)
            
            if ((k%(1000)) == 0) and (k != 0):
                end = timer()
                print(f"sample {k}/{N_samples_train} took:", end-start)
                start = timer()
        end = timer()
        print(f"sample {k}/{N_samples_train} took:", end-start)
        print()
        if self._save:
            self.save_h5(filename=filename)
        
        print("Observing all testing samples:")
        filename = self._filename+'_test'
        
        # Check time dimension of the first sample
        self._result = self.load_qu(filename=filename, k=0)
        self.initialize(N_samples=N_samples_test)
        
        # Take observations
        start = timer()
        for k in range(N_samples_test):
            self._result = self.load_qu(filename=filename, k=k)
            self.observe_one(psi_list=self._result.states, k=k)
            
            if ((k%(1000)) == 0) and (k != 0):
                end = timer()
                print(f"sample {k}/{N_samples_test} took:", end-start)
                start = timer()
        end = timer()
        print(f"sample {k}/{N_samples_test} took:", end-start)
        print()
        if self._save:
            self.save_h5(filename=filename)

    def load_qu(self, filename, k):
        filename = filename+f"_k_{k}.qu"
        self._result = qt.qload(filename)
        self._tlist = self._result.times
        self._Nt = len(self._tlist)
        return self._result

    def delete_qu(self, filename, k):
        filename = filename+f"_k_{k}.qu"
        file_to_rem = Path(filename)
        file_to_rem.unlink(missing_ok=True)

    def delete_qu_all(self):
        N_samples_train = self._N_samples_train
        N_samples_test = self._N_samples_test
        
        print("Deleting all wavefunction training samples:")
        filename = self._filename+'_train'
        start = timer()
        for k in range(N_samples_train):
            self.delete_qu(filename=filename, k=k)
            
            if ((k%(1000)) == 0) and (k != 0):
                end = timer()
                print(f"sample {k}/{N_samples_train} took:", end-start)
                start = timer()
        end = timer()
        print(f"sample {k}/{N_samples_train} took:", end-start)
        print()
        
        print("Deleting all wavefunction testinging samples:")
        filename = self._filename+'_test'
        start = timer()
        for k in range(N_samples_test):
            self.delete_qu(filename=filename, k=k)
            
            if ((k%(1000)) == 0) and (k != 0):
                end = timer()
                print(f"sample {k}/{N_samples_test} took:", end-start)
                start = timer()
        end = timer()
        print(f"sample {k}/{N_samples_test} took:", end-start)
        print()

    def save_h5(self, filename):
        filename = filename+".h5"

        with h5py.File(filename, 'w') as f:
            f.create_dataset('N', data=self._N)
            f.create_dataset('tlist', data=self._tlist)
            
            f.create_dataset(f'x', data=self._x)
            f.create_dataset(f'z', data=self._z)
            f.create_dataset(f'xx', data=self._xx)
            f.create_dataset(f'zz', data=self._zz)
            
            f.create_dataset(f'ee', data=self._ee)
            f.create_dataset(f'es', data=self._es)
            
            f.create_dataset(f'psi', data=self._psi)

    def load_h5(self, filename):
        filename = filename+".h5"

        with h5py.File(filename, 'r') as f:
            self._N = np.array(f['N'])
            self._tlist = np.array( f[f'tlist'] )
            
            self._x = np.array( f[f'x'] )
            self._z = np.array( f[f'z'] )
            self._xx = np.array( f[f'xx'] )
            self._zz = np.array( f[f'zz'] )
            
            self._ee = np.array( f[f'ee'] )
            self._es = np.array( f[f'es'] )
            
            self._psi = np.array( f[f'psi'] )
    
    @property
    def N(self):
        return self._N
    
    @property
    def tlist(self):
        return self._tlist

    @property
    def rho_diag(self):
        return (self._psi.conj() * self._psi).real
    
    @property
    def psi(self):
        return self._psi

    @property
    def z(self):
        return self._z
    
    @property
    def zz(self):
        return self._zz
    
    @property
    def x(self):
        return self._x
    
    @property
    def xx(self):
        return self._xx
    
    @property
    def ee(self):
        return self._ee
    
    @property
    def es(self):
        return self._es
