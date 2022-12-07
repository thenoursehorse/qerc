import numpy as np
import qutip as qt

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

def ising_hamiltonian(N, g, alpha):

    sx_list = get_spin_ops(N=N, axis='x')
    sz_list = get_spin_ops(N=N, axis='z')

    # Set the energy scale
    J = 1

    # For equivalence to time crystal, Akitada would replace
    # g by g = pi/2 * J/0.03 eps

    # construct the hamiltonian
    H = 0

    # magnetic field
    for n in range(N):
        H -= g * sx_list[n]

    # interaction terms
    if alpha == np.inf:
        for i in range(N):
            H += J * sz_list[i] * sz_list[(i+1) % N]

    else:
        for i in range(N):
            for j in range(N):
                if i > j:
                    coupling = J / np.power( np.abs(i-j), alpha)
                    H += coupling * sz_list[i] * sz_list[j]
    return H
