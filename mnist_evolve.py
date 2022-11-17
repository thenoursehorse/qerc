from pathlib import Path
import argparse

from reservoir.qubit_mnist import PCAQubits
from reservoir.evolver import Evolver
from reservoir.observer import Observer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, required=True)
    parser.add_argument('-g', type=float, required=True)
    parser.add_argument('-alpha', type=float, defualt=1.51)
    parser.add_argument('-dt', type=float, defualt=10)
    parser.add_argument('-tf', type=float, defualt=50)
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
    parser.add_argument('-save', type=bool, defualt=True)

    args = parser.parse_args()
    print(args)

    filename = f'{args.model}_N_{args.N}_g_{args.g:0.3f}_alpha_{args.alpha:0.3f}'
    directory = 'data/'+filename
    Path(directory).mkdir(parents=True, exist_ok=True)
    filename = directory+'/'+filename

    # Get pca data of digits
    input_data = PCAQubits(N=args.N)

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
    
    # Measure all samples
    observer = Observer(N=args.N, 
                        filename=filename,
                        N_samples_train=args.N_samples_train,
                        N_samples_test=args.N_samples_test)
    observer.observe_all()
    
    print("Finished.")