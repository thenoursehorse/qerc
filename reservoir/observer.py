import numpy as np
import h5py
from timeit import default_timer as timer

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
                       load=False):
        self._N = N
        self._filename = filename
        
        self._N_samples_train = N_samples_train
        self._N_samples_test = N_samples_test
        
        self._load = load

        self._Nt = None
        self._rho_diag = None

        if self._load:
            self.load_h5(filename=self._filename)

        #self.corr_size = int( (self._N*self._N - self._N)/2 )
        #if not self._load:
        #    self.x_ops = get_spin_ops(N=self._N, axis='x')
        #    self.z_ops = get_spin_ops(N=self._N, axis='z')
        #    self.xx_ops = [self.x_ops[i] * self.x_ops[j] for i in range(self._N) for j in range(self._N) if i > j]
        #    self.zz_ops = [self.z_ops[i] * self.z_ops[j] for i in range(self._N) for j in range(self._N) if i > j]
 
    def initialize(self, N_samples):
        #self.x = np.empty(shape=(self._N, Nt))
        #self.z = np.empty(shape=(self._N, Nt))
        #self.xx = np.empty(shape=(self.corr_size, Nt))
        #self.zz = np.empty(shape=(self.corr_size, Nt))
        #self.ee = np.empty(shape=(Nt))
        #self.es = np.empty(shape=(2**int(self._N/2), Nt))
        self._rho_diag = np.empty(shape=(self._Nt, N_samples, 2**self._N))
        self._psi = np.empty(shape=(self._Nt, N_samples, 2**self._N))

    def time_to_idx(self, time):
        return np.where( np.abs(self._tlist - time) < 1e-10)[0][0]
        
    def observe_one(self, psi_list, k):
        assert isinstance(psi_list, list), "psi must be a list of Qobj !"

        #self.x = np.array(qt.expect(oper=self.x_ops, state=psi_list))
        #self.z = np.array(qt.expect(oper=self.z_ops, state=psi_list))
        #self.xx = np.array(qt.expect(oper=self.xx_ops, state=psi_list))
        #self.zz = np.array(qt.expect(oper=self.zz_ops, state=psi_list))
            
        #zz = observer.zz[:,n]
        #z = observer.z[:,n]
        #x[k,:] = [*zz, *z]

        #for n in range(len(psi_list)):
        #    psi = psi_list[n]
        #    self.es[:,n], self.ee[n] = _entropy(self._N, psi)
   
        for n in range(len(psi_list)):
            psi = psi_list[n]
            self._rho_diag[n,k,:] = (psi.full().conj() * psi.full())[:,0].real
            #if psi.type == 'ket' or psi.type == 'bra':
            #    rho = qt.ket2dm(psi)
            #self._rho_diag[:,n] = np.real(rho.data.diagonal())
            
            self._psi[n,k,:] = psi.full()[:,0]

    def observe_all(self, N_samples_train=None, N_samples_test=None):
        if N_samples_train == None:
            N_samples_train = self._N_samples_train
        if N_samples_test == None:
            N_samples_test = self._N_samples_test
        
        print("Observing all training samples:")
        self.initialize(N_samples=N_samples_train)
        filename = self._filename+'_train'
        start = timer()
        for k in range(N_samples_train):
            result = observer.load_qu(filename=filename, k=k)
            observer.observe_one(psi_list=result.states, k=k)
            
            if ((k%(1000)) == 0) and (k != 0):
                end = timer()
                print(f"sample {k}/{N_samples_train} took:", end-start)
                start = timer()
        end = timer()
        print(f"sample {k}/{N_samples_train} took:", end-start)
        observer.save_h5(filename=filename)
        
        print("Observing all testing samples:")
        self.initialize(N_samples=N_samples_test)
        filename = self._filename+'_test'
        start = timer()
        for k in range(N_samples_test):
            result = observer.load_qu(filename=filename, k=k)
            observer.observe_one(psi_list=result.states, k=k)
            
            if ((k%(1000)) == 0) and (k != 0):
                end = timer()
                print(f"sample {k}/{N_samples_test} took:", end-start)
                start = timer()
        end = timer()
        print(f"sample {k}/{N_samples_test} took:", end-start)
        observer.save_h5(filename=filename)

    def load_qu(self, filename, k=0):
        filename = filename+f"_k_{k}.qu"
        result = qt.qload(filename)
        self._tlist = self.result.times
        self._Nt = len(self._tlist)
        return result

    def save_h5(self, filename):
        filename = filename+".h5"

        with h5py.File(filename, 'w') as f:
            f.create_dataset('N', data=self._N)
            f.create_dataset('tlist', data=self._tlist)
            
            #f.create_dataset(f'x_{k}', data=self.x)
            #f.create_dataset(f'z_{k}', data=self.z)
            
            #f.create_dataset(f'xx_{k}', data=self.xx)
            #f.create_dataset(f'zz_{k}', data=self.zz)
            
            #f.create_dataset(f'ee_{k}', data=self.ee)
            #f.create_dataset(f'es_{k}', data=self.es)
            
            f.create_dataset(f'rho_diag', data=self._rho_diag)
            f.create_dataset(f'psi', data=self._psi)

    def load_h5(self, filename):
        filename = filename+".h5"

        with h5py.File(filename, 'r') as f:
            self._N = f['N']
            self._tlist = np.array( f[f'tlist'] )
            
            #self.x = np.array( f[f'x_{k}'] )
            #self.z = np.array( f[f'z_{k}'] )
            #self.xx = np.array( f[f'xx_{k}'] )
            #self.zz = np.array( f[f'zz_{k}'] )
            
            #self.ee = np.array( f[f'ee_{k}'] )
            #self.es = np.array( f[f'es_{k}'] )
            
            self._rho_diag = np.array( f[f'rho_diag'] )
            self._psi = np.array( f[f'psi'] )

    @property
    def rho_diag(self):
        return self._rho_diag