from pathlib import Path
import argparse
from timeit import default_timer as timer

from reservoir.qubit_mnist import PCAQubits
from elm.elm import ELM
from reservoir.observer import Observer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', type=int, required=True)
    parser.add_argument('-g', type=float, required=True)
    parser.add_argument('-alpha', type=float, defualt=1.51)
    parser.add_argument('-N_samples_train', type=float, default=60000)
    parser.add_argument('-N_samples_test', type=float, default=10000)
    parser.add_argument('-hidden_size', type=int, default=-1)
    parser.add_argument('-activation',
        choices=['softmax', 'sigmoid', 'hyperbolic', 'cos', 'identity'],
        default='softmax',
    )
    parser.add_argument('-random',
        choices=['uniform', 'normal'],
        default='uniform',
    )
    parser.add_argument('-pinv',
        choices=['numpy', 'scipy', 'jax', 'reginv', 'geninv'],
        default='geninv',
    )
    parser.add_argument('-model',
        choices=['ising', 'spin-1/2', 'spin-1'],
        default='ising',
    )
    
    args = parser.parse_args()
    print(args)
    
    # Number of hidden neurons in the elm part
    if args.hidden_size == -1:
        hidden_size = 784
    elif args.hidden_size < -1:
        hidden_size = input_size
    else:
        hidden_size = args.hidden_size
    
    filename = f'{args.model}_N_{args.N}_g_{args.g:0.3f}_alpha_{args.alpha:0.3f}'
    directory = 'data/'+filename
    Path(directory).mkdir(parents=True, exist_ok=True)
    filename = directory+'/'+filename
    
    # Get raw input data of digits
    input_data = PCAQubits(N=args.N)

    # Get x data from reservoir
    filename_train = filename+"_train"
    observer_train = Observer(N=args.N, 
                              filename=filename_train,
                              load=True)
    
    filename_test = filename+"_train"
    observer_test = Observer(N=args.N, 
                             filename=filename_test,
                             load=True)
    
    # This is all samples at every time slice
    x_train_all = observer_train.rho_diag
    x_test_all = observer_test.rho_diag
    #x_train_all = observer_train.psi
    #x_test_all = observer_test.psi
     
    for n in range(1):
        # Take a time slice, indexed as time,sample,inputnode
        x_train = x_train_all[n,:args.N_samples_train,:]
        x_test = x_test_all[n,:args.N_samples_test,:]
        input_size = x_train.shape[-1]
        
        elm = ELM(input_size=input_size, 
                  hidden_size=hidden_size,
                  activation=args.activation,
                  random=args.random,
                  pinv=args.pinv)

        # FIXME it is not clear to me that this makes it better
        # Standardize x
        x_train, x_test = elm.standardize(x_train, x_test)

        # Make the one hot vectors
        y_train, y_test = elm.onehot_y(y_train=input_data.train_y, y_test=input_data.test_y)
     
        # Train the ELM
        start = timer()
        y_train_pred = elm.train(x_train, y_train)
        end = timer()
        print(f"g={args.g} elm took:", end-start)
    
        accuracy_train, mse_train, mae_train = elm.evaluate(y_pred=y_train_pred, y=y_train)
        print("Training mse: ", mse_train)
        print("Training mae: ", mae_train)
        print("Training accuracy: ", accuracy_train)
        print()

        y_test_pred = elm.predict(x_test)
        accuracy_test, mse_test, mae_test = elm.evaluate(y_pred=y_test_pred, y=y_test)
        print("Testing mse: ", mse_test)
        print("Testing mae: ", mae_test)
        print("Testing accuracy: ", accuracy_test)