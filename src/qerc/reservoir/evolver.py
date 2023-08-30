import numpy as np
import scipy
import h5py
from timeit import default_timer as timer

import copy

import qutip as qt

from qerc.reservoir.hamiltonians import *

def _evolve_state(H, psi0, tlist, e_ops=None):
    options = qt.Options(nsteps=4*2500)
    #return qt.sesolve(H, psi0, tlist, options=options)
    #return qt.sesolve(H, psi0, tlist, e_ops, options=options)
    #return qt.mesolve(H=H, rho0=psi0, tlist=tlist, options=options)
    #return qt.mesolve(H=H, rho0=psi0, tlist=tlist, e_ops=e_ops, options=options)
    return qt.mcsolve(H=H, psi0=psi0, tlist=tlist, options=options)
    #return qt.mcsolve(H=H, psi0=psi0, tlist=tlist, e_ops=e_ops, options=options)

def _evolve_state_expm(U, psi0, tlist):
    result = qt.solver.Result()
    result.solver = 'expm'
    result.times = tlist
    states = [qt.Qobj(shape=psi0.shape, dims=psi0.dims) for i in range(len(tlist))]
    states[0] = copy.deepcopy(psi0)
    for i in range(1, len(tlist)):
        states[i] = U * states[i-1]
        #states[i] = qt.Qobj(U.data @ states[i-1].data, dims=states[i-1].dims)
    result.states = states
    return result

def _evolve_state_expm_full(U, psi0, tlist):
    U_full = U.full()
    result = qt.solver.Result()
    result.solver = 'expm'
    result.times = tlist
    states = np.empty(shape=(len(tlist), psi0.shape[0], psi0.shape[1]), dtype=psi0.dtype)
    states[0,...] = copy.deepcopy(psi0)
    for i in range(1, len(tlist)):
        states[i,...] = U_full @ states[i-1,...]
    result.states = states
    return result

def _evolve_state_expm_diag(U_diag, P, psi0, tlist): 
    result = qt.solver.Result()
    result.solver = 'expm_diag'
    result.times = tlist
    states = [qt.Qobj(shape=psi0.shape, dims=psi0.dims) for i in range(len(tlist))]
    states[0] = copy.deepcopy(psi0)
    for i in range(1, len(tlist)):
        # P^+ P expH P^+ P psi0 = P^+ U_diag P psi0
        states[i] = P.dag().data * U_diag * P * states[i-1]
    result.states = states
    return result

def _evolve_state_expm_diag_full(U_diag, P, psi0, tlist):
    U_diag_full = U_diag.full()
    P_full = P.full()
    result = qt.solver.Result()
    result.solver = 'expm_diag'
    result.times = tlist
    states = np.empty(shape=(len(tlist), psi0.shape[0], psi0.shape[1]), dtype=psi0.dtype)
    states[0,...] = copy.deepcopy(psi0)
    for i in range(1, len(tlist)):
        # P^+ P expH P^+ P psi0 = P^+ U_diag P psi0
        states[i,...] = P_full.conj().T @ U_diag_full @ P_full @ states[i-1,...]
    result.states = states
    return result

def _get_U(H, dt):
    ln_U = -1j*H*dt
    U = ln_U.expm()
    return U

def _get_U_diag(eigs, H, dt):
    U_diag = qt.Qobj(dims=H.dims, shape=H.shape)
    di = np.diag_indices(U_diag.shape[0])
    U_diag.data[di] = np.exp(-1j*eigs*dt)
    return U_diag

class Evolver(object):
    '''
    Evolves a collection of initial states psi0 with the unitary 
    U=exp(-i H t). The collection of psi0 may be samples for a machine 
    learning task.

    Args:
        N : Number of lattice sites.

        g : Ising : transverse field. 
            XYZ : Zeeman field.

        alpha : Ising : Power law coupling strength.
                XYZ : ZZ coupling strength.

        filename : Output filename.

        model : (Default 'ising') The type of model to solve. Choices are 
            the transverse field Ising model and spin-1/2 XYZ chain.

        solver : (Default 'expm') Whether the unitary is in the computational 
            basis or is evolved in the diagonal basis. Options are 'expm' or 
            'expm_diag'.

        dt : (Default 1) Time step for solver.

        tf : (Default 5) The final time.

        N_samples_train L: (Default All) Number of train samples to evolve.

        N_samples_test : (Default All) Number of test samples to evolve.

        save : (Default False) Whether to save the wavefunction to a file.
    '''
    def __init__(self, N, 
                       g,
                       alpha,
                       filename,
                       model='ising', 
                       solver='expm', 
                       dt=1, 
                       tf=5, 
                       N_samples_train=None, 
                       N_samples_test=None,
                       save=False):
        self._N = N
        self._g = g
        self._filename = filename

        self._model = model
        self._alpha = alpha
        self._solver = solver
        
        self._dt = dt
        self._tf = tf
        self._tlist = np.arange(0, tf+dt/2.0, dt)

        self._N_samples_train = N_samples_train
        self._N_samples_test = N_samples_test
        
        self._save = save

        self._H = None
        self._eigs = None
        self._vecs = None
        
        self._U = None
        self._U_diag = None
        self._P = None

        self.set_H()
    
    def set_H(self):
        # Construct Hamiltonian
        if self._model == 'ising':
            self._H = ising_hamiltonian(N=self._N, g=self._g, alpha=self._alpha)
        elif self._model == 'xyz':
            self._H = xyz_chain(N=self._N, Delta=self._alpha, g=self._g)
        elif self._model == 'xyz-spin-1':
            raise ValueError(f'{self._model} not implemented yet !')
        else:
            raise ValueError(f'unrecognized model {self._model} !')
    
        # Diagonalize
        self._eigs, self._vecs = self._H.eigenstates()

        # Construct unitaries if required
        if self._solver == "expm":
            self._U = _get_U(H=self._H, dt=self._dt)
   
        if self._solver == "expm_diag":
            self._P = qt.Qobj(dims=self._H.dims, shape=self._H.shape)
            for i in range(self._P.shape[0]):
                self._P.data[i] = self._vecs[i].data.T
            self._U_diag = _get_U_diag(eigs=self._eigs, H=self._H, dt=self._dt)
        
        if self._save:
            qt.qsave(self._H, self._filename+f"_hamiltonian.qu")
            qt.qsave(self._eigs, self._filename+"_eigs.qu")
            qt.qsave(self._vecs, self._filename+"_vecs.qu")
        
    def evolve_single(self, psi0):
        if self._solver == "expm":
            result = _evolve_state_expm_full(U=self._U, psi0=psi0, tlist=self._tlist)
        elif self._solver == "expm_diag":
            result = _evolve_state_expm_diag_full(U_diag=self._U_diag, P=self._P, psi0=psi0, tlist=self._tlist)
        else:
            raise ValueError(f'unrecognized solver {self._solver}')
        return result
    
    def evolve_all(self, psi0_train, psi0_test):
        assert (self._solver == "expm") or (self._solver == "expm_diag"), "Use evolve_all_old for other solvers !"
        
        dims = [ self._H.dims[0], [1 for _ in range(len(self._H.dims[0]))] ]
        shape = (self._H.shape[0], 1)
        
        print("Training:")
        print("Evolving all samples:")
        start = timer()
        result = self.evolve_single(psi0=psi0_train)
        end = timer()
        print("Duration:", end-start)
        print()
        if self._save:
            result.dims = dims
            result.shape = shape
            # evolver indexed as times, basis state, sample, so swap last two indices for observer
            result.states = np.transpose(result.states, (0,2,1))
            qt.qsave(result, self._filename+"_train.qu")
        
        print("Testing:")
        print("Evolving all samples:")
        start = timer()
        result = self.evolve_single(psi0=psi0_test)
        end = timer()
        print(f"Duration:", end-start)
        print()
        if self._save:
            result.dims = dims
            result.shape = shape
            # evolver indexed as times, basis state, sample, so swap last two indices for observer
            result.states = np.transpose(result.states, (0,2,1))
            qt.qsave(result, self._filename+"_test.qu")
    
    def evolve_single_old(self, psi0):
        if self._solver == "mc":
            result = _evolve_state(H=self._H, psi0=psi0, tlist=self._tlist)
        elif self._solver == "expm":
            result = _evolve_state_expm(U=self._U, psi0=psi0, tlist=self._tlist)
        elif self._solver == "expm_diag":
            result = _evolve_state_expm_diag(U_diag=self._U_diag, P=self._P, psi0=psi0, tlist=self._tlist)
        else:
            raise ValueError(f'unrecognized solver {self._solver}')
        return result
    
    def evolve_all_old(self, input_data, N_samples_train=None, N_samples_test=None):
        if N_samples_train == None:
            N_samples_train = self._N_samples_train
        if N_samples_test == None:
            N_samples_test = self._N_samples_test
        
        print("Evolving all training samples:")
        start = timer()
        for k in range(N_samples_train):
            # Encode the digits on psi
            psi0 = input_data.encode_psi(k=k, test=False)

            result = self.evolve_single_old(psi0)

            if self._save:
                qt.qsave(result, self._filename+f"_train_k_{k}.qu")
            
            if ((k%(1000)) == 0) and (k != 0):
                end = timer()
                print(f"sample {k}/{N_samples_train} took:", end-start)
                start = timer()
        end = timer()
        print(f"sample {k}/{N_samples_train} took:", end-start)
        print()
        
        print("Evolving all testing samples:")
        start = timer()
        for k in range(N_samples_test):
            # Encode the digits on psi
            psi0 = input_data.encode_psi(k=k, test=True)

            result = self.evolve_single_old(psi0)

            if self._save:
                qt.qsave(result, self._filename+f"_test_k_{k}.qu")
            
            if ((k%(1000)) == 0) and (k != 0):
                end = timer()
                print(f"sample {k}/{N_samples_test} took:", end-start)
                start = timer()
        end = timer()
        print(f"sample {k}/{N_samples_test} took:", end-start)
        print()
    
    def save(self, filename, result):
        with h5py.File(filename, 'w') as f:
            f.create_dataset('result', data=result)
        
    @property
    def N(self):
        return self._N

    @property
    def g(self):
        return self._g

    @property
    def model(self):
        return self._model
        
    @property
    def tlist(self):
        return self._tlist

    @property
    def H(self):
        return self._H

    @property
    def eigs(self):
        return self._eigs

    @property
    def vecs(self):
        return self._vecs
        
    @property
    def U(self):
        return self._U

    @property
    def U_diag(self):
        return self._U_diag
    
    @property
    def P(self):
        return self._P    