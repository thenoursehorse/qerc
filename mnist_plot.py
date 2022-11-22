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
        choices=['ising', 'spin-1/2', 'spin-1'],
        default='ising',
    )
    parser.add_argument('-node_type',
        choices=['rho_diag', 'psi', 'corr', 'entanglement'],
        default='rho_diag',
    )
    parser.add_argument('-save', type=lambda x: bool(strtobool(x)), default='True')
    parser.add_argument('-show', type=lambda x: bool(strtobool(x)), default='False')

    args = parser.parse_args()
    if args.alpha > 1e3:
        args.alpha = np.inf
    print(args)
        
    if args.save:
        Path('figs/').mkdir(parents=True, exist_ok=True)

    g_list = np.arange(0, 1.3+0.1/2.0, 0.1)
    N_list = np.array([5, 6, 7, 8], dtype=int)

    save_root = 'figs/' + f'{args.model}_alpha_{args.alpha}_node_{args.node_type}_activation_{args.activation}_'
    
    plot = Plotter(g_list=g_list, 
                   N_list=N_list, 
                   model=args.model, 
                   alpha=args.alpha, 
                   node_type=args.node_type, 
                   activation=args.activation, 
                   save_root=save_root, 
                   save=args.save, 
                   show=args.show)

    N = 8
    g = 1.0
    hidden_size = 784
    time = 1.0

    plot.nodes(N=N, g=g, time=time)
    plot.time(N=N, g=g, hidden_size=hidden_size)
    plot.g(N=N, hidden_size=hidden_size, time=time)
    plot.g(hidden_size=hidden_size, time=time)
    plot.N(g=g, hidden_size=hidden_size, time=time)
    
    g_list_for_N = np.array([0.3, 0.6, 0.9, 1.0, 1.1, 1.2])
    plot.N(g_list=g_list_for_N, hidden_size=hidden_size, time=time)