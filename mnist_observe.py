from pathlib import Path
from distutils.util import strtobool
import argparse

from reservoir.observer import Observer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, required=True)
    parser.add_argument('-g', type=float, required=True)
    parser.add_argument('-filename_root', required=True)
    parser.add_argument('-alpha', type=float, default=1.51)
    parser.add_argument('-N_samples_train', type=float, default=60000)
    parser.add_argument('-N_samples_test', type=float, default=10000)
    parser.add_argument('-model',
        choices=['ising', 'spin-1/2', 'spin-1'],
        default='ising',
    )
    parser.add_argument('-delete_qu', type=lambda x: bool(strtobool(x)), default='False')
    parser.add_argument('-save', type=lambda x: bool(strtobool(x)), default='True')

    args = parser.parse_args()
    if args.alpha > 1e3:
        args.alpha = np.inf
    print(args)

    filename = f'{args.model}_N_{args.N}_g_{args.g:0.3f}_alpha_{args.alpha:0.3f}'
    directory = args.filename_root+filename
    Path(directory).mkdir(parents=True, exist_ok=True)
    filename = directory+'/'+filename

    # Measure all samples
    observer = Observer(N=args.N, 
                        filename=filename,
                        N_samples_train=args.N_samples_train,
                        N_samples_test=args.N_samples_test,
                        save=args.save)
    observer.observe_all()

    # Delete all of the qutip wavefunction files
    if args.delete_qu:
        observer.delete_qu_all()
    
    print("Finished.")
