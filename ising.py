import sys
from pathlib import Path
import numpy as np
import h5py
from timeit import default_timer as timer

import copy

import qutip as qt

from mnist_qubit import PCAQubits

def get_spin_ops(N, axis='z'):
    si = qt.qeye(2)
   
    if axis == 'x':
        s = qt.sigmax()
    elif axis == 'y':
        s = qt.sigmay()
    elif axis == 'z':
        s = qt.sigmaz()
    elif axis == '+':
        s = qt.sigmap()
    elif axis == '-':
        s = qt.sigmam()
    else:
        raise ValueError('must be x,y,z,+,- for spin operators')

    s_list = []
    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = s
        s_list.append(qt.tensor(op_list))
    return s_list

def ising_hamiltonian(N, J, g):

    sx_list = get_spin_ops(N=N, axis='x')
    sz_list = get_spin_ops(N=N, axis='z')

    # construct the hamiltonian
    H = 0

    # magnetic field
    for n in range(N):
        H += g * sx_list[n]

    # interaction terms
    #for n in range(N-1):
    #    H += - 0.5 * J * sz_list[n] * sz_list[n+1]
    
    # interaction terms
    alpha = 1.51
    for i in range(N):
        for j in range(N):
            if i != j:
                coupling = J / np.power( np.abs(i-j), alpha)
                H += coupling * sz_list[i] * sz_list[j]
    return H

def evolve_state(H, psi0, tlist, e_ops=None):
    options = qt.Options(nsteps=4*2500)
    #return qt.sesolve(H, psi0, tlist, options=options)
    #return qt.sesolve(H, psi0, tlist, e_ops, options=options)
    #return qt.mesolve(H=H, rho0=psi0, tlist=tlist, options=options)
    #return qt.mesolve(H=H, rho0=psi0, tlist=tlist, e_ops=e_ops, options=options)
    return qt.mcsolve(H=H, psi0=psi0, tlist=tlist, options=options)
    #return qt.mcsolve(H=H, psi0=psi0, tlist=tlist, e_ops=e_ops, options=options)

def evolve_state_expm(U, psi0, tlist): 
    output = qt.solver.Result()
    output.solver = 'expm'
    output.times = tlist
    states = [None for i in range(len(tlist))]
    states[0] = copy.deepcopy(psi0)
    for i in range(1, len(tlist)):
        states[i] = U * states[i-1]
    output.states = states
    return output

def evolve_state_expm_diag(U_diag, P, psi0, tlist): 
    output = qt.solver.Result()
    output.solver = 'expm_diag'
    output.times = tlist
    states = [None for i in range(len(tlist))]
    states[0] = copy.deepcopy(psi0)
    for i in range(1, len(tlist)):
        # P^+ P expH P^+ P psi0 = P^+ U_diag P psi0
        states[i] = P.dag() * U_diag * P * states[i-1]
    output.states = states
    return output

def get_U(H, dt):
    ln_U = -1j*H*dt
    U = ln_U.expm()
    return U

def get_U_diag(eigs, H, dt):
    U_diag = qt.Qobj(dims=H.dims, shape=H.shape)
    di = np.diag_indices(U_diag.shape[0])
    U_diag.data[di] = np.exp(-1j*eigs*dt)
    return U_diag

if __name__ == '__main__':
    # The parameters
    print("Usage: python3 ising.py <N> <g> <N_samples> <solver> <test>")
    if len(sys.argv) != 6:
        sys.exit(1)
    print('Argument List:', str(sys.argv))
    N = int(sys.argv[1])
    g = float(sys.argv[2])
    N_samples = int(sys.argv[3])
    solver = str(sys.argv[4])
    test = str(sys.argv[5])
    if test in ['True','true','1']:
        test = True
    else:
        test = False
   
    if not test:
        #filename = f'data/ising_train_N_{N}_g_{g:0.3f}.h5'
        directory = f'data/ising_train_N_{N}_g_{g:0.3f}'
        filename = f'ising_train_N_{N}_g_{g:0.3f}'
    else:
        #filename = f'data/ising_test_N_{N}_g_{g:0.3f}.h5'
        directory = f'data/ising_test_N_{N}_g_{g:0.3f}'
        filename = f'ising_test_N_{N}_g_{g:0.3f}'
    Path(directory).mkdir(parents=True, exist_ok=True)

    # set energy scale
    J = 1
    
    # times for evolution and setup observer to measure at each timestep
    #tlist = np.linspace(0, 50, 501)
    #tlist = np.linspace(0, 50, 11)
    dt = 10
    tlist = np.arange(0, 50+dt/2.0, dt)

    # Construct Hamiltonian
    start = timer()
    hamiltonian = ising_hamiltonian(N=N, J=J, g=g)
    end = timer()
    print("Time to construct Hamiltonian:", end-start)
    qt.qsave(hamiltonian, directory+f"/{filename}_hamiltonian.qu")
    
    start = timer()
    eigs, vecs = hamiltonian.eigenstates()
    end = timer()
    print("Time to diagonalize Hamiltonian:", end-start)
    qt.qsave(eigs, directory+f"/{filename}_eigs.qu")
    qt.qsave(vecs, directory+f"/{filename}_vecs.qu")
    
    if solver == "expm":
        start = timer()
        U = get_U(H=hamiltonian, dt=dt)
        end = timer()
        print("Time to construct U:", end-start)
   
    elif solver == "expm_diag":
        start = timer()
        P = qt.Qobj(dims=hamiltonian.dims, shape=hamiltonian.shape)
        for i in range(P.shape[0]):
            P.data[i] = vecs[i].data.T
        end = timer()
        print("Time to construct P:", end-start)
        qt.qsave(P, directory+f"/{filename}_P.qu")
        
        start = timer()
        U_diag = get_U_diag(eigs=eigs, H=hamiltonian, dt=dt)
        end = timer()
        print("Time to construct U_diag:", end-start)

    # Get pca data of digits
    pca = PCAQubits(N=N)
    if not test:
        if N_samples <= 0:
            N_samples = pca.train_theta.shape[0]
        else:
            assert N_samples <= pca.train_theta.shape[0], f"N_samples={N_samples} must be <= {pca.train_theta.shape[0]}"
    else:
        if N_samples <= 0:
            N_samples = pca.test_theta.shape[0]
        else:
            assert N_samples <= pca.test_theta.shape[0], f"N_samples={N_samples} must be <= {pca.test_theta.shape[0]}"
    
    # Big ol loop
    start_outer = timer()
    for k in range(N_samples):
        
        # Encode the digits on psi
        if not test:
            psi0 = pca.encode_psi(theta=pca.train_theta[k], phi=pca.train_phi[k])
        else:
            psi0 = pca.encode_psi(theta=pca.test_theta[k], phi=pca.test_phi[k])

        start = timer()
        if solver == "mc":
            ##result = evolve_state(H=hamiltonian, psi0=psi0, tlist=tlist, e_ops=observer.obs)
            result = evolve_state(H=hamiltonian, psi0=psi0, tlist=tlist)
        elif solver == "expm":
            result = evolve_state_expm(U=U, psi0=psi0, tlist=tlist)
        elif solver == "expm_diag":
            result = evolve_state_expm_diag(U_diag=U_diag, P=P, psi0=psi0, tlist=tlist)
        end = timer()
        print(f"g={g} test={test}. Time to evolve sample {k+1}/{N_samples}:", end-start)
        print()

        qt.qsave(result, directory+f"/{filename}_k_{k}.qu")
    
    end_outer = timer()
    print(f"Time to evolve all samples:", end_outer-start_outer)

    print("Finished.")
