import sys
import numpy as np
import h5py
from timeit import default_timer as timer

import qutip as qt

from ising import get_spin_ops

def vn_spectrum(rho, base=np.e, sparse=False):
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

def get_entropy(N, psi, sparse=False):
    #half_ind = [0]
    half_ind = [i for i in range(int(N/2))]
    
    rhoA = psi.ptrace(half_ind, sparse=sparse)
    
    vals, logvals = vn_spectrum(rhoA)

    #entropy_entanglement = qt.entropy.entropy_vn(rhoA, sparse=sparse)
    # mutual is 2*entropy_entanglement for pure states
    #rho = qt.states.ket2dm(psi)
    #half_compliment_ind = [half_ind[i]+int(N/2) for i in range(int(N/2))]
    #half_compliment_ind = [i+1 for i in range(N-1)]
    #entropy_mutual = qt.entropy.entropy_mutual(rho, half_ind, half_compliment_ind, sparse=sparse)
    return vals, float(np.real(-sum(vals[vals != 0] * logvals)))

class Observer:
    def __init__(self, N, tlist=None, filename='data/ising.h5', load=False):
        self.N = N
        self.filename = filename
        self.corr_size = int( (self.N*self.N - self.N)/2 )
        self.load = load
             
        if not self.load:
            self.x_ops = get_spin_ops(N=self.N, axis='x')
            self.z_ops = get_spin_ops(N=self.N, axis='z')
            self.xx_ops = [self.x_ops[i] * self.x_ops[j] for i in range(self.N) for j in range(self.N) if i > j]
            self.zz_ops = [self.z_ops[i] * self.z_ops[j] for i in range(self.N) for j in range(self.N) if i > j]
 
    def initialize(self):
        self.x = np.empty(shape=(self.N, self.Nt))
        self.z = np.empty(shape=(self.N, self.Nt))
        self.xx = np.empty(shape=(self.corr_size, self.Nt))
        self.zz = np.empty(shape=(self.corr_size, self.Nt))
        self.ee = np.empty(shape=(self.Nt))
        self.es = np.empty(shape=(2**int(self.N/2), self.Nt))
        self.rho_diag = np.empty(shape=(2**self.N, self.Nt))

    def time_to_idx(self, time):
        #n = self.time_to_idx(time)
        return np.where( np.abs(self.tlist - time) < 1e-10)[0][0]
        
    def obs(self, psi_list=None):
        if psi_list == None:
            psi_list = self.result.states

        assert isinstance(psi_list, list), "psi must be a list of Qobj !"
        assert len(psi_list) == self.Nt, f"psi must be a list at different times of length {self.Nt} !"

        #self.x = np.array(qt.expect(oper=self.x_ops, state=psi_list))
        #self.z = np.array(qt.expect(oper=self.z_ops, state=psi_list))
        #self.xx = np.array(qt.expect(oper=self.xx_ops, state=psi_list))
        #self.zz = np.array(qt.expect(oper=self.zz_ops, state=psi_list))

        #for n in range(self.Nt):
        #    psi = psi_list[n]
        #    self.es[:,n], self.ee[n] = get_entropy(self.N, psi)
   
        for n in range(self.Nt):
            psi = psi_list[n]
            self.rho_diag[:,n] = (psi.full().conj() * psi.full())[:,0].real
            #if psi.type == 'ket' or psi.type == 'bra':
            #    rho = qt.ket2dm(psi)
            #self.rho_diag[:,n] = np.real(rho.data.diagonal())

    def load_qu(self, k=0, filename=None):
        if filename == None:
            filename = self.filename+f"_k_{k}.qu"
        self.result = qt.qload(filename)
        self.tlist = self.result.times
        self.Nt = len(self.tlist)
        self.initialize()

    def save_h5(self, k=0, filename=None):
        if filename == None:
            filename = self.filename+".h5"

        if k == 0:
            with h5py.File(filename, 'w') as f:
                f.create_dataset('N', data=self.N)
                f.create_dataset('Nt', data=self.Nt)
                f.create_dataset('tlist', data=self.tlist)
            
        with h5py.File(filename, 'a') as f:
            #f.create_dataset(f'x_{k}', data=self.x)
            #f.create_dataset(f'z_{k}', data=self.z)
            
            #f.create_dataset(f'xx_{k}', data=self.xx)
            #f.create_dataset(f'zz_{k}', data=self.zz)
            
            #f.create_dataset(f'ee_{k}', data=self.ee)
            #f.create_dataset(f'es_{k}', data=self.es)
            
            f.create_dataset(f'rho_diag_{k}', data=self.rho_diag)

    def load_h5(self, k=0, filename=None):
        if filename == None:
            filename = self.filename+".h5"

        with h5py.File(filename, 'r') as f:
            self.N = f['N']
            self.Nt = f['Nt']
            self.tlist = np.array( f[f'tlist'] )
            
            #self.x = np.array( f[f'x_{k}'] )
            #self.z = np.array( f[f'z_{k}'] )
            #self.xx = np.array( f[f'xx_{k}'] )
            #self.zz = np.array( f[f'zz_{k}'] )
            
            #self.ee = np.array( f[f'ee_{k}'] )
            #self.es = np.array( f[f'es_{k}'] )
            
            self.rho_diag = np.array( f[f'rho_diag_{k}'] )

if __name__ == '__main__':
    # The parameters
    print("Usage: python3 observer.py <N> <g> <N_samples> <test>")
    if len(sys.argv) != 5:
        sys.exit(1)
    print('Argument List:', str(sys.argv))
    N = int(sys.argv[1])
    g = float(sys.argv[2])
    N_samples = int(sys.argv[3])
    test = str(sys.argv[4])
    if test in ['True','true','1']:
        test = True
    else:
        test = False

    if not test:
        #filename = f'data/ising_train_N_{N}_g_{g:0.3f}.h5'
        directory = f'data/ising_train_N_{N}_g_{g:0.3f}'
        filename = f'ising_train_N_{N}_g_{g:0.3f}'
        if N_samples < 0:
            N_samples = 60000
    else:
        #filename = f'data/ising_test_N_{N}_g_{g:0.3f}.h5'
        directory = f'data/ising_test_N_{N}_g_{g:0.3f}'
        filename = f'ising_test_N_{N}_g_{g:0.3f}'
        if N_samples < 0:
            N_samples = 10000

    # uniform couplings between sites and on-site potential
    J = 1
    
    filename = directory+"/"+filename

    observer = Observer(N=N, filename=filename)

    for k in range(N_samples):
        #start = timer()
        observer.load_qu(k=k)
        #end = timer()
        #print(f"g={g} test={test}. Time to load {k+1}/{N_samples}:", end-start)
        
        start = timer()
        observer.obs()
        end = timer()
        print(f"g={g} test={test}. Time to take obs of {k+1}/{N_samples}:", end-start)
        
        #start = timer()
        observer.save_h5(k=k)
        #end = timer()
        #print(f"g={g} test={test}. Time to save h5:", end-start)
        
        print()
