import numpy as np
import h5py
from timeit import default_timer as timer
from pathlib import Path

import qutip as qt

from qerc.reservoir.hamiltonians import get_spin_ops

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

class Observer(object):
    '''
    Evolves a collection of initial states psi0 with the unitary 
    U=exp(-i H t). The collection of psi0 may be samples for a machine 
    learning task.

    Args:
        N : Number of lattice sites.

        filename : Output filename from Evolver.

        N_samples_train L: (Default All) Number of train samples to evolve.

        N_samples_test : (Default All) Number of test samples to evolve.

        save : (Default True) Whether to save the observations to a hdf5 file.

        load : (Default False) Whether to load a previously generated hdf5 
            file of observables.

        observe_list : (Default ["psi"]) A list of strings that are 
            observables to calculate. Options are
            psi : Coefficients of psi in the computational basis.
            x : Expectation <S_i^x> at each site i.
            xx : Correlation <S_i^x S_j^x> between all sites i and j.
            z : Expectation <S_i^z> at each site i.
            zz : Correlation <S_i^z S_j^z> between all sites i and j.
            ee : von-Nuemann entangelement entropy. Bipartition is at center 
                site.
            es : Entanglement spectrum. Bipartition is at center site.
    '''
    def __init__(self, N,
                       filename, 
                       N_samples_train=None, 
                       N_samples_test=None,
                       save=True,
                       load=False,
                       observe_list=["psi"]):
                       #observe_list=["psi","x","xx","z","zz","ee","es"]):
        self._N = N
        self._filename = filename
        self._filename_train = self._filename+"_train"
        self._filename_test = self._filename+"_test"
        
        self._N_samples_train = N_samples_train
        self._N_samples_test = N_samples_test
        
        self._save = save
        self._load = load

        self._observe_list = observe_list

        self._result = None
        self._Nt = None

        if self._load:
            self.load_h5(filename=self._filename)

        self._corr_size = int( (self._N*self._N - self._N)/2 )
        if not self._load:
            if "x" in self._observe_list:
                self._x_ops = get_spin_ops(N=self._N, axis='x')
            if "z" in self._observe_list:
                self._z_ops = get_spin_ops(N=self._N, axis='z')
            if "xx" in self._observe_list:
                self._xx_ops = [self._x_ops[i] * self._x_ops[j] for i in range(self._N) for j in range(self._N) if i > j]
            if "zz" in self._observe_list:
                self._zz_ops = [self._z_ops[i] * self._z_ops[j] for i in range(self._N) for j in range(self._N) if i > j]
 
    def initialize(self, N_samples):
        if "x" in self._observe_list:
            self._x = np.empty(shape=(self._Nt, N_samples, self._N))
        if "z" in self._observe_list:
            self._z = np.empty(shape=(self._Nt, N_samples, self._N))
        if "xx" in self._observe_list:
            self._xx = np.empty(shape=(self._Nt, N_samples, self._corr_size))
        if "zz" in self._observe_list:
            self._zz = np.empty(shape=(self._Nt, N_samples, self._corr_size))
        if "ee" in self._observe_list:
            self._ee = np.empty(shape=(self._Nt, N_samples))
        if "es" in self._observe_list:
            self._es = np.empty(shape=(self._Nt, N_samples, 2**int(self._N/2)))
        if "psi" in self._observe_list:
            self._psi = np.empty(shape=(self._Nt, N_samples, 2**self._N), dtype=complex)

    def time_to_idx(self, time):
        return np.where( np.abs(self._tlist - time) < 1e-10)[0][0]
    
    def _to_Qobj(self, result, k):
        states = result.states[:,k,:]
        dims = result.dims
        shape = result.shape
        Nt = len(states)
        return [qt.Qobj(result.states[t,k,:], dims=dims, shape=shape) for t in range(Nt)]
    
    def observe_one(self, result, k):
        # Make into Qobj to take observations
        psi_list = self._to_Qobj(result=result, k=k)

        # qt.expect returns indexed as obs, time, hence transpose
        if "x" in self._observe_list:
            self._x[:,k,:] = np.array(qt.expect(oper=self._x_ops, state=psi_list)).T
        if "z" in self._observe_list:
            self._z[:,k,:] = np.array(qt.expect(oper=self._z_ops, state=psi_list)).T
        if "xx" in self._observe_list:
            self._xx[:,k,:] = np.array(qt.expect(oper=self._xx_ops, state=psi_list)).T
        if "zz" in self._observe_list:
            self._zz[:,k,:] = np.array(qt.expect(oper=self._zz_ops, state=psi_list)).T
            
        if ("ee" in self._observe_list) or ("es" in self._observe_list):
            for n in range(len(psi_list)):
                psi = psi_list[n]
                self._es[n,k,:], self._ee[n,k] = _entropy(self._N, psi)

    def observe_all(self):
        N_samples_train = self._N_samples_train
        N_samples_test = self._N_samples_test
        
        print("Observing all training samples:")
        self._result = self.load_qu(filename=self._filename_train)
        self.initialize(N_samples=N_samples_train)
        
        if "psi" in self._observe_list:
            self._psi[...] = self._result.states[...]
        
        if ("x" or "xx" or "z" or "zz" or "ee" or "es") in self._observe_list:
            # Take observations
            start = timer()
            for k in range(N_samples_train):
                self.observe_one(result=self._result, k=k)
                
                if ((k%(1000)) == 0) and (k != 0):
                    end = timer()
                    print(f"sample {k}/{N_samples_train} took:", end-start)
                    start = timer()
            end = timer()
            print(f"sample {k}/{N_samples_train} took:", end-start)
            print()
        
        if self._save:
            self.save_h5(filename=self._filename_train)
            
        print("Observing all testing samples:")
        self._result = self.load_qu(filename=self._filename_test)
        self.initialize(N_samples=N_samples_test)
            
        if "psi" in self._observe_list:
           self._psi[...] = self._result.states[...]
            
        if ("x" or "xx" or "z" or "zz" or "ee" or "es") in self._observe_list:
            # Take observations
            start = timer()
            for k in range(N_samples_test):
                self.observe_one(result=self._result, k=k)
                
                if ((k%(1000)) == 0) and (k != 0):
                    end = timer()
                    print(f"sample {k}/{N_samples_test} took:", end-start)
                    start = timer()
            end = timer()
            print(f"sample {k}/{N_samples_test} took:", end-start)
            print()
        
        if self._save:
            self.save_h5(filename=self._filename_test)

    def observe_one_old(self, psi_list, k):
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

    def observe_all_old(self):
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
            self.observe_one_old(psi_list=self._result.states, k=k)
            
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
            self.observe_one_old(psi_list=self._result.states, k=k)
            
            if ((k%(1000)) == 0) and (k != 0):
                end = timer()
                print(f"sample {k}/{N_samples_test} took:", end-start)
                start = timer()
        end = timer()
        print(f"sample {k}/{N_samples_test} took:", end-start)
        print()
        if self._save:
            self.save_h5(filename=filename)

    def load_qu(self, filename, k=None):
        if k != None:
            filename = filename+f"_k_{k}.qu"
        else:
            filename = filename+".qu"
        self._result = qt.qload(filename)
        self._tlist = self._result.times
        self._Nt = len(self._tlist)
        return self._result

    def delete_qu(self, filename, k=None):
        if k != None:
            filename = filename+f"_k_{k}.qu"
        else:
            filename = filename+".qu"
        file_to_rem = Path(filename)
        file_to_rem.unlink(missing_ok=True)

    def delete_qu_all(self):
        
        print("Deleting all wavefunction training samples:")
        self.delete_qu(filename=self._filename_train)
        
        print("Deleting all wavefunction testing samples:")
        self.delete_qu(filename=self._filename_test)

    def delete_qu_all_old(self):
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
        
        print("Deleting all wavefunction testing samples:")
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
            
            if "x" in self._observe_list:
                f.create_dataset(f'x', data=self._x)
            if "z" in self._observe_list:
                f.create_dataset(f'z', data=self._z)
            if "xx" in self._observe_list:
                f.create_dataset(f'xx', data=self._xx)
            if "zz" in self._observe_list:
                f.create_dataset(f'zz', data=self._zz)
            
            if "ee" in self._observe_list:
                f.create_dataset(f'ee', data=self._ee)
            if "es" in self._observe_list:
                f.create_dataset(f'es', data=self._es)
            
            if "psi" in self._observe_list:
                f.create_dataset(f'psi', data=self._psi)

    def load_h5(self, filename):
        filename = filename+".h5"

        with h5py.File(filename, 'r') as f:
            self._N = np.array(f['N'])
            self._tlist = np.array( f[f'tlist'] )
            
            if "x" in self._observe_list:
                self._x = np.array( f[f'x'] )
            if "z" in self._observe_list:
                self._z = np.array( f[f'z'] )
            if "xx" in self._observe_list:
                self._xx = np.array( f[f'xx'] )
            if "zz" in self._observe_list:
                self._zz = np.array( f[f'zz'] )
            
            if "ee" in self._observe_list:
                self._ee = np.array( f[f'ee'] )
            if "es" in self._observe_list:
                self._es = np.array( f[f'es'] )
            
            if "psi" in self._observe_list:
                self._psi = np.array( f[f'psi'] )
    
    @property
    def N(self):
        return self._N
    
    @property
    def tlist(self):
        return self._tlist

    @property
    def rho_diag(self):
        assert "psi" in self._observe_list, "psi must be loaded in class to calculate rho_diag !"
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
