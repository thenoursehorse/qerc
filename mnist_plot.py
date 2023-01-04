from pathlib import Path
from distutils.util import strtobool
import argparse

import numpy as np

from analyze.plotter import Plotter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-alpha', type=float, default=1.51)
    parser.add_argument('-activation',
        choices=['softmax', 'sigmoid', 'hyperbolic', 'cos', 'identity'],
        default='softmax',
    )
    parser.add_argument('-model',
        choices=['ising', 'xyz'],
        default='ising',
    )
    parser.add_argument('-node_type',
        choices=['rho_diag', 'psi', 'corr', 'entanglement'],
        default='rho_diag',
    )
    parser.add_argument('-nn_type',
        choices=['perceptron', 'identity', 'elm', 'mlp'],
        default='perceptron',
    )
    parser.add_argument('-save', type=lambda x: bool(strtobool(x)), default='True')
    parser.add_argument('-show', type=lambda x: bool(strtobool(x)), default='False')

    args = parser.parse_args()
    if args.alpha > 1e3:
        args.alpha = np.inf
    print(args)
        
    if args.save:
        Path('figs/').mkdir(parents=True, exist_ok=True)

    g_list = []
    g_list = np.append(g_list, np.arange(0.5, 5+0.5/2.0, 0.5))
    g_list = np.append(g_list, np.arange(0.1, 2+0.1/2.0, 0.1))
    g_list = np.append(g_list, np.arange(0.9, 1.1+0.01/2.0, 0.01))
    g_list = np.unique(g_list)

    N_list = np.array([5, 6, 7, 8, 9, 10, 11], dtype=int)
    
    save_root = 'figs/' + f'{args.model}_alpha_{args.alpha}_node_{args.node_type}_activation_{args.activation}_'
    
    plot = Plotter(g_list=g_list, 
                   N_list=N_list, 
                   model=args.model, 
                   alpha=args.alpha, 
                   node_type=args.node_type,
                   nn_type=args.nn_type,
                   activation=args.activation, 
                   save_root=save_root, 
                   save=args.save, 
                   show=args.show)
    
    #if args.nn_type == 'elm':
    #    plot.nodes(N=N, g=g, time=time)

    if args.nn_type == 'perceptron':
        plot.epochs(N=10, g=0.5, time=5)
        plot.epochs(N=10, g=1, time=5)
        plot.epochs(N=10, g=5, time=5)

    plot.time(N=10, g=5)
    plot.time(N=11, g=5)
    
    plot.time(N=10, g=1)
    plot.time(N=11, g=1)
    
    xmin = 0.8 #0
    xmax = 1.2 #np.max(g_list)
    plot.g(N=10, time=1, xmin=xmin, xmax=xmax)
    plot.g(N=10, time=1.5, xmin=xmin, xmax=xmax)
    plot.g(N=10, time=2, xmin=xmin, xmax=xmax)
    plot.g(N=10, time=2.5, xmin=xmin, xmax=xmax)
    plot.g(N=10, time=3, xmin=xmin, xmax=xmax)
    plot.g(N=10, time=3.5, xmin=xmin, xmax=xmax)
    plot.g(N=10, time=4, xmin=xmin, xmax=xmax)
    plot.g(N=10, time=4.5, xmin=xmin, xmax=xmax)
    plot.g(N=10, time=5, xmin=xmin, xmax=xmax)
    
    plot.g(N=11, time=1, xmin=xmin, xmax=xmax)
    plot.g(N=11, time=1.5, xmin=xmin, xmax=xmax)
    plot.g(N=11, time=2, xmin=xmin, xmax=xmax)
    plot.g(N=11, time=2.5, xmin=xmin, xmax=xmax)
    plot.g(N=11, time=3, xmin=xmin, xmax=xmax)
    plot.g(N=11, time=3.5, xmin=xmin, xmax=xmax)
    plot.g(N=11, time=4, xmin=xmin, xmax=xmax)
    plot.g(N=11, time=4.5, xmin=xmin, xmax=xmax)
    plot.g(N=11, time=5, xmin=xmin, xmax=xmax)

    plot.g(N_list=[7,8,9,10,11], time=1, xmin=xmin, xmax=xmax)
    plot.g(N_list=[7,8,9,10,11], time=1.5, xmin=xmin, xmax=xmax)
    plot.g(N_list=[7,8,9,10,11], time=2, xmin=xmin, xmax=xmax)
    plot.g(N_list=[7,8,9,10,11], time=2.5, xmin=xmin, xmax=xmax)
    plot.g(N_list=[7,8,9,10,11], time=3, xmin=xmin, xmax=xmax)
    plot.g(N_list=[7,8,9,10,11], time=3.5, xmin=xmin, xmax=xmax)
    plot.g(N_list=[7,8,9,10,11], time=4, xmin=xmin, xmax=xmax)
    plot.g(N_list=[7,8,9,10,11], time=4.5, xmin=xmin, xmax=xmax)
    plot.g(N_list=[7,8,9,10,11], time=5, xmin=xmin, xmax=xmax)
    
    plot.N(g=1, time=5)
    
    g_list_for_N = np.array([0.1, 1, 3, 5])
    plot.N(g_list=g_list_for_N, time=5)