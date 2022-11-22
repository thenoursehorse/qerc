from pathlib import Path
import argparse
from distutils.util import strtobool
from timeit import default_timer as timer

import h5py
import numpy as np

from reservoir.qubit_mnist import PCAQubits
from elm.elm import ELM
from reservoir.observer import Observer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, required=True)
    parser.add_argument('-g', type=float, required=True)
    parser.add_argument('-filename_root', required=True)
    parser.add_argument('-alpha', type=float, default=1.51)
    parser.add_argument('-N_samples_train', type=float, default=60000)
    parser.add_argument('-N_samples_test', type=float, default=10000)
    parser.add_argument('-hsize_initial', type=int, default=-1)
    parser.add_argument('-hsize_final', type=int, default=1000)
    parser.add_argument('-hsize_step', type=int, default=500)
    parser.add_argument('-activation',
        choices=['softmax', 'sigmoid', 'hyperbolic', 'cos', 'identity'],
        default='softmax',
    )
    parser.add_argument('-random',
        choices=['uniform', 'normal'],
        default='uniform',
    )
    parser.add_argument('-pinv',
        choices=['numpy', 'scipy', 'jax', 'reginv', 'IMqrginv', 'geninv'],
        default='geninv',
    )
    parser.add_argument('-model',
        choices=['ising', 'spin-1/2', 'spin-1'],
        default='ising',
    )
    parser.add_argument('-node_type',
        choices=['rho_diag', 'psi', 'corr', 'entanglement'],
        default='rho_diag',
    )
    parser.add_argument('-standardize', type=lambda x: bool(strtobool(x)), default='False')
    parser.add_argument('-save', type=lambda x: bool(strtobool(x)), default='True')
    
    args = parser.parse_args()
    if args.alpha > 1e3:
        args.alpha = np.inf
    print(args)
     
    filename = f'{args.model}_N_{args.N}_g_{args.g:0.3f}_alpha_{args.alpha:0.3f}'
    directory = args.filename_root+filename
    Path(directory).mkdir(parents=True, exist_ok=True)
    filename = directory+'/'+filename
    
    # Get raw input data of digits
    input_data = PCAQubits(N=args.N, filename=args.filename_root)

    # Get x data from reservoir
    filename_train = filename+"_train"
    observer_train = Observer(N=args.N, 
                              filename=filename_train,
                              load=True)
    
    filename_test = filename+"_test"
    observer_test = Observer(N=args.N, 
                             filename=filename_test,
                             load=True)
    
    # Below are all samples at every time slice
    tlist_train = observer_train.tlist
    tlist_test = observer_test.tlist
    
    assert (tlist_train == tlist_test), 'training and testing must have same times !'
    tlist = tlist_train

    # Measuring only in computational basis
    if args.node_type == 'rho_diag':
        x_train_all = observer_train.rho_diag
        x_test_all = observer_test.rho_diag
    
    # Full wavefunction information
    elif args.node_type == 'psi':
        shape = observer_train.psi.shape
        x_train_all = np.empty(shape=(shape[0], shape[1], 2*shape[2]))
        x_train_all[:,:,:shape[2]] = observer_train.psi.real
        x_train_all[:,:,shape[2]:] = observer_train.psi.imag
    
        shape = observer_test.psi.shape
        x_test_all = np.empty(shape=(shape[0], shape[1], 2*shape[2]))
        x_test_all[:,:,:shape[2]] = observer_test.psi.real
        x_test_all[:,:,shape[2]:] = observer_test.psi.imag

    elif args.node_type == 'corr':
        z_shape = observer_train.z.shape
        zz_shape = observer_train.zz.shape
        x_train_all = np.empty(shape=(z_shape[0], z_shape[1], z_shape[2]+zz_shape[2]))
        x_train_all[:,:,:z_shape[2]] = observer_train.z
        x_train_all[:,:,z_shape[2]:] = observer_train.zz
        
        z_shape = observer_test.z.shape
        zz_shape = observer_test.zz.shape
        x_test_all = np.empty(shape=(z_shape[0], z_shape[1], z_shape[2]+zz_shape[2]))
        x_test_all[:,:,:z_shape[2]] = observer_test.z
        x_test_all[:,:,z_shape[2]:] = observer_test.zz

    elif args.node_type == 'entanglement':
        x_train_all = observer_train.es
        x_test_all = observer_test.es

    else:
        raise ValueError('unrecognized x input node type !')

    assert (x_train_all.shape[0] == x_test_all.shape[0]), 'training and testing time dimension differ !'
    assert (x_train_all.shape[-1] == x_test_all.shape[-1]), 'training and testing number of input nodes differ !'
     
    # Number of input nuerons
    input_size = x_train_all.shape[-1]
    
    # Number of hidden neurons in the elm part
    if args.activation == 'identity':
        hidden_array = [0]
    elif args.hsize_initial == -1:
        hidden_array = np.array([784])
    elif args.hsize_initial < -1:
        hidden_array = np.array([input_size])
    else:
        hidden_array = np.arange(args.hsize_initial, args.hsize_final + args.hsize_step/2.0, args.hsize_step, dtype=int)
        hidden_array = np.append(hidden_array, [input_size, 784])
        hidden_array.sort()

    # Number of time slices
    Nt = x_train_all.shape[0]
   
    accuracy_train = np.empty(shape=(len(hidden_array), Nt))
    accuracy_test = np.empty(shape=(len(hidden_array), Nt))
    mse_train = np.empty(shape=(len(hidden_array), Nt))
    mse_test = np.empty(shape=(len(hidden_array), Nt))
    mae_train = np.empty(shape=(len(hidden_array), Nt))
    mae_test = np.empty(shape=(len(hidden_array), Nt))
    
    for h in range(len(hidden_array)):
        hidden_size = hidden_array[h]
        print(f"ELM for hidden size = {hidden_size}")
        for n in range(Nt):
            # Take a time slice, indexed as time,sample,inputnode
            x_train = x_train_all[n,:args.N_samples_train,:]
            x_test = x_test_all[n,:args.N_samples_test,:]
            
            elm = ELM(input_size=input_size, 
                      hidden_size=hidden_size,
                      activation=args.activation,
                      random=args.random,
                      pinv=args.pinv)

            # FIXME standardizing x actually makes it worse and for some
            # activation functions it is much much much worse
            # Standardize x
            if args.standardize:
                x_train, x_test = elm.standardize(x_train, x_test)

            # Make the one hot vectors
            y_train, y_test = elm.onehot_y(y_train=input_data.train_y, y_test=input_data.test_y)
         
            # Train the ELM
            start = timer()
            try:
                y_train_pred = elm.train(x_train, y_train)
            except Exception as e:
                print(e)
                print(f"WARNING: elm training with {args.pinv} failed. Falling back to IMqrginv.")
                elm._pinv = 'IMqrginv'
                y_train_pred = elm.train(x_train, y_train)
            end = timer()
            print(f"n = {n+1}/{Nt} elm took:", end-start)
        
            accuracy_train[h,n], mse_train[h,n], mae_train[h,n] = elm.evaluate(y_pred=y_train_pred, y=y_train)
            print("Training accuracy: ", accuracy_train[h,n])

            y_test_pred = elm.predict(x_test)
            accuracy_test[h,n], mse_test[h,n], mae_test[h,n] = elm.evaluate(y_pred=y_test_pred, y=y_test)
            print("Testing accuracy: ", accuracy_test[h,n])
            print()

    filename_elm = filename+"_elm"
    filename_elm += f"_nodetype_{args.node_type}"
    filename_elm += f"_activation_{args.activation}"
    filename_elm += ".h5"
    if args.save:
        with h5py.File(filename_elm, 'w') as f:
            f.create_dataset('tlist', data=tlist)
            f.create_dataset('hidden_array', data=hidden_array)
            f.create_dataset('accuracy_train', data=accuracy_train)
            f.create_dataset('accuracy_test', data=accuracy_test)
            f.create_dataset('mse_train', data=mse_train)
            f.create_dataset('mse_test', data=mse_test)
            f.create_dataset('mae_train', data=mae_train)
            f.create_dataset('mae_test', data=mae_test)