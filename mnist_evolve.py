from pathlib import Path
from distutils.util import strtobool
import argparse

import numpy as np

from encoder.qubit_mnist import PCAQubits
from reservoir.evolver import Evolver

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, required=True)
    parser.add_argument('-g', type=float, required=True)
    parser.add_argument('-filename_root', required=True)
    parser.add_argument('-alpha', type=float, default=1.51)
    parser.add_argument('-dt', type=float, default=10)
    parser.add_argument('-tf', type=float, default=50)
    parser.add_argument('-N_samples_train', type=float, default=60000)
    parser.add_argument('-N_samples_test', type=float, default=10000)
    parser.add_argument('-solver',
        choices=['mc', 'expm', 'expm_diag'],
        default='expm',
    )
    parser.add_argument('-model',
        choices=['ising', 'spin-1/2', 'spin-1'],
        default='ising',
    )
    parser.add_argument('-save', type=lambda x: bool(strtobool(x)), default='True')

    args = parser.parse_args()
    if args.alpha > 1e3:
        args.alpha = np.inf
    print(args)

    filename = f'{args.model}_N_{args.N}_g_{args.g:0.3f}_alpha_{args.alpha:0.3f}'
    directory = args.filename_root+filename
    Path(directory).mkdir(parents=True, exist_ok=True)
    filename = directory+'/'+filename

    # Get pca data of digits
    input_data = PCAQubits(N=args.N, filename=args.filename_root)

    # Evolve all samples
    evolver = Evolver(N=args.N, 
                      g=args.g,
                      filename=filename,
                      model=args.model, 
                      solver=args.solver, 
                      dt=args.dt, 
                      tf=args.tf, 
                      alpha=args.alpha, 
                      N_samples_train=args.N_samples_train, 
                      N_samples_test=args.N_samples_test,
                      save=args.save)
    evolver.evolve_all(input_data=input_data)
    
    print("Finished.")
