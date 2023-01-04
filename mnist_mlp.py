from pathlib import Path
import argparse
from distutils.util import strtobool
from timeit import default_timer as timer

import numpy as np

from encoder.qubit_mnist import PCAQubits
from nnetwork.mlp import MLP
from reservoir.observer import Observer

if __name__ == '__main__':
    print("Ising: g tranvserse field. alpha power law coupling.")
    print("XYZ: g Zeeman field. alpha ZZ coupling.")
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, required=True)
    parser.add_argument('-g', type=float, required=True)
    parser.add_argument('-alpha', type=float, required=True)
    parser.add_argument('-model',
        choices=['ising', 'xyz'],
        required=True,
    )
    parser.add_argument('-filename_root', required=True)
    parser.add_argument('-N_samples_train', type=float, default=60000)
    parser.add_argument('-N_samples_test', type=float, default=10000)
    parser.add_argument('-node_type',
        choices=['rho_diag', 'psi', 'corr', 'entanglement'],
        default='rho_diag',
    )
    parser.add_argument('-N_epochs', type=int, default=10)
    parser.add_argument('-stats_stride', type=int, default=10)
    parser.add_argument('-M', type=int, default=128)
    parser.add_argument('-eta', type=float, default=0.01)
    parser.add_argument('-shuffle', type=lambda x: bool(strtobool(x)), default='False')
    parser.add_argument('-standardize', type=lambda x: bool(strtobool(x)), default='True')
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
    
    assert (tlist_train == tlist_test).all(), 'training and testing must have same times !'
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
    
    # Number of time slices
    Nt = x_train_all.shape[0]

    # There are 10 categories for MNIST
    output_size = 10

    accuracy_train = np.empty(shape=(args.N_epochs+1, Nt))
    accuracy_test = np.empty(shape=(args.N_epochs+1, Nt))
    mse_train = np.empty(shape=(args.N_epochs+1, Nt))
    mse_test = np.empty(shape=(args.N_epochs+1, Nt))
    mae_train = np.empty(shape=(args.N_epochs+1, Nt))
    mae_test = np.empty(shape=(args.N_epochs+1, Nt))
    losses_train = np.empty(shape=(args.N_epochs+1, Nt))
    losses_test = np.empty(shape=(args.N_epochs+1, Nt))

    avg_train = np.empty(shape=(Nt))
    std_train = np.empty(shape=(Nt))
    best_train = np.empty(shape=(Nt))
    avg_test = np.empty(shape=(Nt))
    std_test = np.empty(shape=(Nt))
    best_test = np.empty(shape=(Nt))
    
    for n in range(Nt):
        # Take a time slice, indexed as time,sample,inputnode
        x_train = x_train_all[n,:args.N_samples_train,:]
        x_test = x_test_all[n,:args.N_samples_test,:]
            
        mlp = MLP(input_size=input_size, 
                  output_size=output_size,
                  N_epochs=args.N_epochs,
                  M=args.M,
                  eta=args.eta,
                  shuffle=args.shuffle)

        # Standardize x
        if args.standardize:
            x_train, x_test = mlp.standardize(x_train, x_test)

        # Make the one hot vectors
        y_train, y_test = mlp.onehot_y(y_train=input_data.y_train, y_test=input_data.y_test)
         
        # Train deep network and take some obvservations
        start = timer()
        y_train_pred, y_test_pred = mlp.train(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
        end = timer()
        print(f"n = {n+1}/{Nt} mlp took:", end-start)

        accuracy_train[:,n] = mlp.accuracy_train 
        losses_train[:,n] = mlp.losses_train
        avg_train[n] = np.mean(accuracy_train[-args.stats_stride:-1,n])
        std_train[n] = np.std(accuracy_train[-args.stats_stride:-1,n])
        best_train[n] = np.max(accuracy_train[:,n])
        print("Training accuracy mean: ", avg_train[n])
        print("Training accuracy std: ", std_train[n])
        print("Training best: ", best_train[n])

        accuracy_test[:,n] = mlp.accuracy_test
        losses_test[:,n] = mlp.losses_test
        avg_test[n] = np.mean(accuracy_test[-args.stats_stride:-1,n])
        std_test[n] = np.std(accuracy_test[-args.stats_stride:-1,n])
        best_test[n] = np.max(accuracy_test[:,n])
        print("Testing accuracy mean: ", avg_test[n])
        print("Testing accuracy std: ", std_test[n])
        print("Testing best: ", best_test[n])
        print()
        
    if args.save:
        from nnetwork.nndata import NNData
            
        filename_perc = filename+"_mlp"
        filename_perc += f"_nodetype_{args.node_type}"
        filename_perc += f"_activation_softmax"
        filename_perc += ".h5"

        nndata = NNData(filename=filename_perc,
                        tlist=tlist,
                        accuracy_train=accuracy_train,
                        accuracy_test=accuracy_test,
                        losses_train=losses_train,
                        losses_test=losses_test,
                        avg_train=avg_train,
                        avg_test=avg_test,
                        std_train=std_train,
                        std_test=std_test,
                        best_train=best_train,
                        best_test=best_test,
                        N=args.N,
                        g=args.g,
                        alpha=args.alpha,
                        N_samples_train=args.N_samples_train,
                        N_samples_test=args.N_samples_test,
                        activation="softmax",
                        model=args.model,
                        node_type=args.node_type,
                        standardize=args.standardize,
                        N_epochs=args.N_epochs,
                        stats_stride=args.stats_stride,
                        M=args.M,
                        eta=args.eta)
        nndata.save() 