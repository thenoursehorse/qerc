from pathlib import Path
from distutils.util import strtobool
import argparse

import numpy as np

from qerc.analyze.plotter import Plotter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model',
        choices=['ising', 'xyz'],
        required=True,
    )
    parser.add_argument('-activation',
        choices=['softmax', 'sigmoid', 'hyperbolic', 'cos', 'identity'],
        default='softmax',
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
    print(args)
        
    if args.save:
        Path('figs/').mkdir(parents=True, exist_ok=True)

    if args.model == 'ising':
        N_list = np.array([5, 6, 7, 8, 9, 10, 11], dtype=int)
        
        g_list = []
        g_list = np.append(g_list, np.arange(0.1, 5.0+0.1/2.0, 0.1))
        g_list = np.append(g_list, np.arange(0.9, 1.1+0.01/2.0, 0.01))
        g_list = np.unique(g_list)
            
        g_list_zoom = [0.8]
        g_list_zoom = np.append(g_list_zoom, np.arange(0.9, 1.1+0.01/2.0, 0.01))
        g_list_zoom = np.append(g_list_zoom, [1.2])
        g_list_zoom = np.unique(g_list_zoom)
        
        for alpha in [1.51, np.inf]:        
            save_root = 'figs/' 
            save_root += f'model_{args.model}/'
            save_root += f'alpha_{alpha}/'
            save_root += f'nn_type_{args.nn_type}/'
            save_root += f'node_type_{args.node_type}/'
            save_root += f'activation_{args.activation}/'
            Path(save_root).mkdir(parents=True, exist_ok=True)
            
            plot = Plotter(g_list=g_list, 
                           N_list=N_list, 
                           model=args.model, 
                           alpha=alpha, 
                           node_type=args.node_type,
                           nn_type=args.nn_type,
                           activation=args.activation, 
                           save_root=save_root, 
                           save=args.save, 
                           show=args.show)
            
            #if args.nn_type == 'elm':
            #    plot.nodes(N=N, g=g, time=time)

            if args.nn_type == 'perceptron':
                plot.epochs(N=10, g=0.4, time=5)
                plot.epochs(N=10, g=1, time=5)
                plot.epochs(N=10, g=5, time=5)

            plot.time(N=10, g=5)
            plot.time(N=11, g=5)
            
            plot.time(N=10, g=1)
            plot.time(N=11, g=1)
            
            plot.time(N=10, g=0)
            plot.time(N=11, g=0)
            
            xmin = 0
            xmax = np.max(g_list)
            ymin = 0.9
            ymax = 1.0
            for t in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]:
                plot.g(N=10, time=t, xmin=xmin, xmax=xmax)
                plot.g(N=11, time=t, xmin=xmin, xmax=xmax)
                plot.g(N_list=[7,8,9,10,11], time=t, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
            
            plot.N(g=1, time=5)
            
            g_list_for_N = np.array([0.1, 1, 3, 5])
            plot.N(g_list=g_list_for_N, time=5)

            # Do zoomed in
            save_root = 'figs/' 
            save_root += f'model_{args.model}_zoom/'
            save_root += f'alpha_{alpha}/'
            save_root += f'nn_type_{args.nn_type}/'
            save_root += f'node_type_{args.node_type}/'
            save_root += f'activation_{args.activation}/'
            Path(save_root).mkdir(parents=True, exist_ok=True)
            
            plot = Plotter(g_list=g_list_zoom, 
                           N_list=N_list, 
                           model=args.model, 
                           alpha=alpha, 
                           node_type=args.node_type,
                           nn_type=args.nn_type,
                           activation=args.activation, 
                           save_root=save_root, 
                           save=args.save, 
                           show=args.show)
            
            xmin = np.min(g_list_zoom)
            xmax = np.max(g_list_zoom)
            ymin = 0.9
            ymax = 1.0
            for t in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]:
                plot.g(N=10, time=t, xmin=xmin, xmax=xmax)
                plot.g(N=11, time=t, xmin=xmin, xmax=xmax)
                plot.g(N_list=[7,8,9,10,11], time=t, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    
    elif args.model == 'xyz':
        N_list = np.array([5, 6, 7, 8, 9, 10, 11], dtype=int)

        # HACK because chose bad naming convention
        # g is Delta (J^z diff from Jx,Jy)
        # alpha is g^z, a mag field in the z direction 
        g_list = np.arange(-2.0, 2+0.1/2.0, 0.1)
        alpha = 0
        
        save_root = 'figs/' 
        save_root += f'model_{args.model}/'
        save_root += f'g_{alpha}/'
        save_root += f'nn_type_{args.nn_type}/'
        save_root += f'node_type_{args.node_type}/'
        save_root += f'activation_{args.activation}/'
        Path(save_root).mkdir(parents=True, exist_ok=True)
            
        plot = Plotter(g_list=g_list, 
                       N_list=N_list, 
                       model=args.model, 
                       alpha=alpha, 
                       node_type=args.node_type,
                       nn_type=args.nn_type,
                       activation=args.activation, 
                       save_root=save_root, 
                       save=args.save, 
                       show=args.show)

        if args.nn_type == 'perceptron':
            plot.epochs(N=10, g=-1, time=5)
            plot.epochs(N=10, g=0, time=5)
            plot.epochs(N=10, g=1, time=5)
            plot.epochs(N=10, g=2, time=5)

        plot.time(N=10, g=0)
        plot.time(N=11, g=0)
            
        plot.time(N=10, g=1)
        plot.time(N=11, g=1)
        
        plot.time(N=10, g=2)
        plot.time(N=11, g=2)
            
        xmin = np.min(g_list)
        xmax = np.max(g_list)
        ymin = 0.9
        ymax = 1.0
        for t in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]:
            plot.g(N=10, time=t, xmin=xmin, xmax=xmax)
            plot.g(N=11, time=t, xmin=xmin, xmax=xmax)
            plot.g(N_list=[7,8,9,10,11], time=t, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
            
        plot.N(g=-1, time=5)
        plot.N(g=0, time=5)
        plot.N(g=1, time=5)
        plot.N(g=2, time=5)
            
        g_list_for_N = np.array([-1, 0, 1, 2])
        plot.N(g_list=g_list_for_N, time=5)
    
    else:
        raise ValueError("Unrecognized model !")