import numpy as np
from scipy.optimize import curve_fit

import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import cm
        
from qerc.classifier.nndata import NNData

plt.rcParams.update({
    #'text.usetex' : True
    #'mathtext.fontset' : 'stix',
    'mathtext.fontset' : 'cm',
    'font.family' : 'STIXGeneral',
    'lines.markersize' : 2,
    'lines.linewidth' : 1,
})

# From https://stackoverflow.com/a/59543988
class FigureWrapper(object):
    '''Frees underlying figure when it goes out of scope. 
    '''
    def __init__(self, figure):
        self._figure = figure

    def __del__(self):
        plt.close(self._figure)

class Plotter(object):
    def __init__(self, g_list, 
                       N_list, 
                       alpha, 
                       model, 
                       node_type='rho_diag', 
                       activation='softmax', 
                       filename_root = 'data/', 
                       column_width=3.4, 
                       full_column_width=7.0,
                       save=True,
                       show=False,
                       save_root='figs/',
                       nn_type='perceptron'
                       ):
        self._N_list = N_list
        self._g_list = g_list
        self._alpha = alpha
        self._model = model
        self._node_type = node_type
        self._activation = activation
        self._filename_root = filename_root
        self._nn_type = nn_type
        
        self._column_width = column_width
        self._full_column_width = full_column_width

        self._save = save
        self._show = show
        self._save_root = save_root

        self._clip_on = True

    def load(self):
        # HACK because I did stupid naming conventions
        if self._model == 'ising':
            g = self._g
            alpha = self._alpha
        elif self._model == 'xyz':
            g = self._alpha
            alpha = self._g
        else:
            raise ValueError("Unrecognized model !")

        filename_params = f'{self._model}_N_{self._N}_g_{g:0.3f}_alpha_{alpha:0.3f}'
        filename = self._filename_root
        filename += filename_params+"/"+filename_params
        filename += "_" + self._nn_type
        filename += f"_nodetype_{self._node_type}"
        filename += f"_activation_{self._activation}"
        filename += ".h5"
        
        data = NNData(filename=filename)
        data.load()
        
        self._tlist = data.tlist
        self._mse_train = data.mse_train
        self._mse_test = data.mse_test
        self._mae_train = data.mae_train
        self._mae_test = data.mae_test
        
        #if self._nn_type == 'elm':
        #    self._hidden_array = data.hidden_array
        
        if self._nn_type == 'perceptron':
            self._accuracy_train = data.accuracy_train
            self._accuracy_test = data.accuracy_test
           
            self._avg_train = data.avg_train
            self._avg_test = data.avg_test
            self._std_train = data.std_train
            self._std_test = data.std_test
            
            #self._losses_train = data.losses_train
            #self._losses_test = data.losses_train

            self._N_epochs = data.N_epochs

        elif (self._nn_type == 'elm') and (self._activation == 'identity'):
            self._accuracy_train = data.accuracy_train
            self._accuracy_test = data.accuracy_test

            self._avg_train = self._accuracy_train[0,:]
            self._avg_test = self._accuracy_test[0,:]
            self._std_train = np.zeros(shape=self._avg_train.shape)
            self._std_test = np.zeros(shape=self._avg_test.shape)

        else:
            raise ValueError(f"Unrecognized neural network type !")

    def power_func(self, x, a, b, c):
        return a * np.power(x, b) + c

    def log_func(self, x, a, b):
        return a * np.log2(x) + b

    def logistic_func(self, x, a, b, c, d):
        return a / (1.0 + np.exp(-b*(x-c))) + d

    def _hidden_value_to_index(self, value):
        idx = np.where(value == self._hidden_array)[0]
        if idx.size == 0:
            raise ValueError(f"{value} not in hidden_array !")
        return idx[0]
    
    def _time_value_to_index(self, value):
        idx = np.where(value == self._tlist)[0]
        if idx.size == 0:
            raise ValueError(f"{value} not in tlist !")
        return idx[0]

    def _data_picker(self, data):
        if data == 'Training':
            accuracy = self._accuracy_train
            mse = self._mse_train
            mae = self._mae_train
            avg = self._avg_train
            std = self._std_train

        elif data == 'Testing':
            accuracy = self._accuracy_test
            mse = self._mse_test
            mae = self._mae_test
            avg = self._avg_test
            std = self._std_test

        else:
            raise ValueError(f'unknown data type {data}.')
        
        return accuracy, avg, std, mse, mae
        
    def _nodes_picker(self, hidden_size, N):
        if hidden_size < 0:
            if self._node_type == 'rho_diag':
                nodes = 2**N
            elif self._node_type == 'psi':
                nodes = 2 * 2**N
            elif self._node_type == 'corr':
                nodes = int( (N*N - N)/2 ) + N
            elif self._node_type == 'entanglement':
                nodes = 2**int(N/2)
            else:
                raise ValueError(f'Unrecognized node type for picking number of nodes in elm !')
        else:
            nodes = hidden_size
        return nodes

    def _axis_limits(self, x, y, xmin, xmax, ymin, ymax):
        #if xmin == None:
        #    xmin = np.min(x)
        #if xmax == None:
        #    xmax = np.max(x)
        #if ymin == None:
        #    ymin = np.min(y)
        #if ymax == None:
        #    ymax = np.max(y)

        return xmin, xmax, ymin, ymax

    def _get_figure(self, N_figs=1, gap_height_ratio=0.2):
        num_gaps = N_figs - 1                           # for vertical stacking
        
        canvas_width = self._column_width * 0.7          # fixed size area we draw on
        gap_width = canvas_width * 0.075                 # For the right part of figure
        x_start_width = canvas_width * 0.25              # for label and axis
        fig_width = x_start_width + canvas_width + gap_width
        
        x_start = x_start_width / fig_width
        x_span = canvas_width / fig_width
        
        canvas_height = canvas_width * 0.7              # fixed size area we draw on
        title_height = canvas_height * 0.3              # for super title
        gap_height = canvas_height * gap_height_ratio   # for axis (and maybe label)
        y_start_height = canvas_height * 0.3            # for label and axis
        fig_height = N_figs*canvas_height + num_gaps*gap_height + y_start_height + title_height
        
        title_span = title_height / fig_height
        y_start = y_start_height / fig_height
        gap_span = gap_height / fig_height
        y_span = canvas_height / fig_height

        fig = plt.figure(figsize=(self._column_width, fig_height))
        # add_axes(left, bottom, width, height) as fractions of fig height and width
        axis = []
        for i in range(N_figs):
            y = y_start + i*y_span + i*gap_span
            axis.append( fig.add_axes([x_start, y, x_span, y_span]) )
        axis = list(reversed(axis))

        for ax in axis:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        
        return fig, axis

    # From https://github.com/gka/chroma.js
    def _color_picker(self, N_lines):
        assert (N_lines > 0), 'Number of lines must be greater than zero !'

        if N_lines == 1:
            return ['#73a2c6']
        
        elif N_lines == 2:
            return ['#00429d', '#93003a']

        elif N_lines == 3:
            return ['#00429d', '#909074', '#93003a']
        
        elif N_lines == 4:
            return ['#00429d', '#3f69ae', '#f4777f', '#93003a']
        
        elif N_lines == 5:
            return ['#00429d', '#3f69ae', '#909074', '#f4777f', '#93003a']
        
        elif N_lines == 6:
            return ['#00429d', '#315ca9', '#4c77b3', '#ffa59e', '#dd4c65', '#93003a']
        
        elif N_lines == 7:
            return ['#00429d', '#315ca9', '#4c77b3', '#909074', '#ffa59e', '#dd4c65', '#93003a']

        elif N_lines == 8:
            return ['#00429d', '#2955a6', '#3f69ae', '#527eb5', '#ffbcaf', '#f4777f', '#cf3759', '#93003a']
        
        elif N_lines == 9:
            return ['#00429d', '#2955a6', '#3f69ae', '#527eb5', '#909074', '#ffbcaf', '#f4777f', '#cf3759', '#93003a']
        
        elif N_lines == 10:
            return ['#00429d', '#2451a4', '#3761ab', '#4772b1', '#5682b6', '#ffcab9', '#fd9291', '#e75d6f', '#c52a52', '#93003a']

        else:
            raise ValueError(f'Do you really need to put more than 10 lines on a figure?')

    def _marker_picker(self, N_lines):
        assert (N_lines), 'Number of lines must be greater than zero !'
        
        if N_lines == 1:
            return ['o']

        elif N_lines == 2:
            return ['o', 'v']
            
        elif N_lines == 3:
            return ['o', 'v', '^']
            
        elif N_lines == 4:
            return ['o', 'v', '^', '<']
            
        elif N_lines == 5:
            return ['o', 'v', '^', '<', '>']
            
        elif N_lines == 6:
            return ['o', 'v', '^', '<', '>', 's']
            
        elif N_lines == 7:
            return ['o', 'v', '^', '<', '>', 's', '*']
            
        elif N_lines == 8:
            return ['o', 'v', '^', '<', '>', 's', '*', 'x']
            
        elif N_lines == 9:
            return ['o', 'v', '^', '<', '>', 's', '*', 'x', 'D', '+']
            
        else:
            raise ValueError(f'Do you really need to put more than 10 lines on a figure?')

    def nodes(self, N, g, time=1.0, xmin=None, xmax=None, ymin=None, ymax=None):
        assert self._nn_type == 'elm', 'Plotting scaling with nodes requires the elm classical optimizer'

        color = self._color_picker(N_lines=2)
        marker = self._marker_picker(N_lines=2)
        fig, axis = self._get_figure(N_figs=2)
        _wrapped_figure = FigureWrapper(fig)

        self._N = N
        self._g = g
        self.load()
        x = self._hidden_array
        n = self._time_value_to_index(time)

        # Ignore small values
        idx = np.where(self._hidden_array >= 400)[0]
        x_lin = np.linspace(1e-6, np.max(x), 1000)

        for i, data in enumerate(['Training', 'Testing']):
            accuracy, mse, mae = self._data_picker(data=data)

            # FIXME curve fitting, what fucntion actually fits here?
            #popt, pcov = curve_fit(self.log_func, x[idx], accuracy[idx,n])
            #axis[0].plot(x_lin, self.log_func(x_lin, *popt), color=color[i])
            #accuracy_max = self.log_func(10000, *popt)
            #axis[0].plot([0, np.max(x)], [accuracy_max, accuracy_max], '--', color=color[i])
            
            axis[0].plot(x, accuracy[:,n], linestyle='None', color=color[i], marker=marker[i], label=data, clip_on=self._clip_on, zorder=10)
            axis[0].plot(x, accuracy[:,n], '-', color=color[i], clip_on=self._clip_on, zorder=10)
            axis[1].plot(x, mse[:,n], linestyle='None', color=color[i], marker=marker[i], clip_on=self._clip_on, zorder=10)
            axis[1].plot(x, mse[:,n], '-', color=color[i], clip_on=self._clip_on, zorder=10)
            #axis[2].plot(x, mae[:,n], linestyle='None', color=color[i], marker=marker[i], clip_on=False, zorder=10)
            #axis[2].plot(x, mae[:,n], '-', color=color[i], clip_on=False, zorder=10)

            #mse_min_idx = np.argmin(mse[:,n])
            #mse_min = accuracy[mse_min_idx,n]
            #axis[0].plot([0, np.max(x)], [mse_min, mse_min], '--', color=color[i])
        
        xmin, xmax, ymin, ymax = self._axis_limits(x=x, y=y, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        
        axis[0].set_ylabel('Acc.')
        axis[0].set_ylim(ymin=ymin, ymax=ymax)
        axis[0].set_xlim(xmin=xmin, xmax=xmax)
        axis[0].legend(frameon=False)
        
        axis[1].set_ylabel('Mean-squared error')
        axis[1].set_ylim(ymin=ymin, ymax=ymax)
        axis[1].set_xlim(xmin=xmin, xmax=xmax)
        
        #axis[2].set_ylabel('Absolute-squared error')
        #axis[2].set_ylim(ymin=0)
        #axis[2].set_xlim(xmin=0, xmax=np.max(x))
        axis[1].set_xlabel('Hidden nodes')

        title = f'N = {N}     g = {g}     $\\alpha$ = {self._alpha}     t = {time}'
        fig.suptitle(title)

        if self._save:
            filename = f"nodes_N_{self._N}_g_{self._g}_t_{time}"
            plt.savefig(self._save_root+filename+'.pdf')

        if self._show:
            plt.show()

    def epochs(self, N, g, time=1.0, xmin=0.0, xmax=None, ymin=None, ymax=1.0):
        assert self._nn_type == 'perceptron', 'Plotting scaling with epochs requires the perceptron classical optimizer'
        
        color = self._color_picker(N_lines=2)
        marker = self._marker_picker(N_lines=2)
        fig, axis = self._get_figure(N_figs=1)
        _wrapped_figure = FigureWrapper(fig)

        self._N = N
        self._g = g
        self.load()
        x = np.array( [i+1 for i in range(1, self._N_epochs+1)] ) # ignore the 0th epoch
        
        n = self._time_value_to_index(time)
        for i, data in enumerate(['Training', 'Testing']):
            accuracy, avg, std, mse, mae = self._data_picker(data=data)
            y = accuracy[1:,n,0] # ignore the 0th epoch, only take 1 realiziation to show trend
            
            axis[0].plot(x, y, linestyle='None', clip_on=self._clip_on, label=data, zorder = 10, color=color[i], marker=marker[i])
            axis[0].plot(x, y, '-', clip_on=self._clip_on, zorder = 10, color=color[i])
    
        xmin, xmax, ymin, ymax = self._axis_limits(x=x, y=y, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        
        axis[0].set_ylabel('Acc.')
        axis[0].set_ylim(ymin=ymin, ymax=ymax)
        axis[0].set_xlim(xmin=xmin, xmax=xmax)
        axis[0].legend(frameon=False)
        axis[0].set_xlabel('Epoch')
        
        if self._model == 'ising':
            title = f'{self._model}     N = {N}     g = {g}     $\\alpha$ = {self._alpha}     time = {time}'
        elif self._model == 'xyz':
            title = f'{self._model}     N = {N}     $\\Delta$ = {g}     time = {time}'
        else:
            raise ValueError("Unrecognized model !")
        fig.suptitle(title)
        
        if self._save:
            if self._model == 'ising':
                filename = f"epochs_N_{self._N}_g_{self._g}_t_{time}"
            elif self._model == 'xyz':
                filename = f"epochs_N_{self._N}_Delta_{self._g}_t_{time}"
            else:
                raise ValueError("Unrecognized model !")
            plt.savefig(self._save_root+filename+'.pdf')

        if self._show:
            plt.show()
    
    def time(self, N, g, xmin=0.0, xmax=None, ymin=None, ymax=1.0):
        color = self._color_picker(N_lines=2)
        marker = self._marker_picker(N_lines=2)
        fig, axis = self._get_figure(N_figs=1)
        _wrapped_figure = FigureWrapper(fig)

        self._N = N
        self._g = g
        self.load()
        x = self._tlist
        
        for i, data in enumerate(['Training', 'Testing']):
            accuracy, avg, std, mse, mae = self._data_picker(data=data)
            y = avg
            y_err = std
            
            axis[0].plot(x, y, linestyle='None', clip_on=self._clip_on, label=data, zorder = 10, color=color[i], marker=marker[i])
            axis[0].plot(x, y, '-', clip_on=self._clip_on, zorder = 10, color=color[i])
            axis[0].fill_between(x, y-y_err, y+y_err, alpha=0.5, antialiased=True, color=color[i], linewidth=0.0)
        
        xmin, xmax, ymin, ymax = self._axis_limits(x=x, y=y, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        
        axis[0].set_ylabel('Acc.')
        axis[0].set_ylim(ymin=ymin, ymax=ymax)
        axis[0].set_xlim(xmin=xmin, xmax=xmax)
        axis[0].legend(frameon=False)
        axis[0].set_xlabel(r'$t J$')
        
        if self._model == 'ising':
            title = f'{self._model}     N = {N}     g = {g}     $\\alpha$ = {self._alpha}'
        elif self._model == 'xyz':
            title = f'{self._model}     N = {N}     $\\Delta$ = {g}'
        else:
            raise ValueError("Unrecognized model !")
        fig.suptitle(title)
        
        if self._save:
            if self._model == 'ising':
                filename = f"time_N_{self._N}_g_{self._g}"
            elif self._model == 'xyz':
                filename = f"time_N_{self._N}_Delta_{self._g}"
            else:
                raise ValueError("Unrecognized model !")
            plt.savefig(self._save_root+filename+'.pdf')

        if self._show:
            plt.show()

    def _N_fixed_g(self, g, N_list=None, time=1.0, xmin=None, xmax=None, ymin=None, ymax=None):
        if N_list is None:
            N_list = self._N_list
        
        color = self._color_picker(N_lines=2)
        marker = self._marker_picker(N_lines=2)
        fig, axis = self._get_figure(N_figs=1)
        _wrapped_figure = FigureWrapper(fig)

        x = 1.0/N_list
        self._g = g
        
        for i, data in enumerate(['Training', 'Testing']):
            y = np.empty(shape=len(x))
            y_err = np.empty(shape=len(x))
            for j, self._N in enumerate(N_list):
                self.load()
                n = self._time_value_to_index(time)
                
                accuracy, avg, std, mse, mae = self._data_picker(data=data)
                y[j] = avg[n]
                y_err[j] = std[n]
            
            axis[0].plot(x, y, linestyle='None', clip_on=self._clip_on, color=color[i], marker=marker[i], label=data, zorder=10)
            axis[0].plot(x, y, '-', clip_on=self._clip_on, color=color[i], zorder=10)
            axis[0].fill_between(x, y-y_err, y+y_err, alpha=0.5, antialiased=True, color=color[i], linewidth=0.0)
        
        xmin, xmax, ymin, ymax = self._axis_limits(x=x, y=y, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        
        axis[0].set_xlabel(r'$1/N$')
        axis[0].set_ylabel('Acc.')
        axis[0].set_ylim(ymin=ymin, ymax=ymax)
        axis[0].set_xlim(xmin=xmin, xmax=xmax)
        axis[0].legend(frameon=False)
        
        if self._model == 'ising':
            title = f'{self._model}     g = {g}     $\\alpha$ = {self._alpha}     t = {time}'
        elif self._model == 'xyz':
            title = f'{self._model}     $\\Delta$ = {g}     t = {time}'
        else:
            raise ValueError("Unrecognized model !")
        fig.suptitle(title)
        
        if self._save:
            if self._model == 'ising':
                filename = f"scaleN_g_{self._g}_t_{time}"
            elif self._model == 'xyz':
                filename = f"scaleN_Delta_{self._g}_t_{time}"
            else:
                raise ValueError("Unrecognized model !")
            plt.savefig(self._save_root+filename+'.pdf')

        if self._show:
            plt.show()
    
    def _N_all(self, g_list=None, N_list=None, time=1.0, xmin=None, xmax=None, ymin=None, ymax=None):
        if g_list is None:
            g_list = self._g_list
        if N_list is None:
            N_list = self._N_list

        color = self._color_picker(N_lines=len(g_list))
        marker = self._marker_picker(N_lines=len(g_list))
        fig, axis = self._get_figure(N_figs=2)
        _wrapped_figure = FigureWrapper(fig)
        
        x = 1.0/N_list

        for i, data in enumerate(['Training', 'Testing']):
            for k, self._g in enumerate(g_list):
                y = np.empty(shape=len(x))
                y_err = np.empty(shape=len(x))
                for j, self._N in enumerate(N_list):
                    self.load()
                    n = self._time_value_to_index(time)
                    
                    accuracy, avg, std, mse, mae = self._data_picker(data=data)
                    y[j] = avg[n]
                    y_err[j] = std[n]

                axis[i].plot(x, y, linestyle='None', clip_on=self._clip_on, label=f'g = {self._g:.2f}', zorder = 10, color=color[k], marker=marker[k])
                axis[i].plot(x, y, '-', clip_on=self._clip_on, zorder = 10, color=color[k])
                axis[i].fill_between(x, y-y_err, y+y_err, alpha=0.5, antialiased=True, color=color[k], linewidth=0.0)
        
        xmin, xmax, ymin, ymax = self._axis_limits(x=x, y=y, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        
        axis[0].set_ylabel('Training Acc.')
        axis[0].set_ylim(ymin=ymin, ymax=ymax)
        axis[0].set_xlim(xmin=xmin, xmax=xmax)
        
        axis[1].set_xlabel(r'$1/N$')
        axis[1].set_ylabel('Testing Acc.')
        axis[1].set_ylim(ymin=ymin, ymax=ymax)
        axis[1].set_xlim(xmin=xmin, xmax=xmax)
        axis[1].legend(frameon=False)
        
        if self._model == 'ising':
            title = f'{self._model}     $\\alpha$ = {self._alpha}     t = {time}'
        elif self._model == 'xyz':
            title = f'{self._model}     t = {time}'
        else:
            raise ValueError("Unrecognized model !")
        fig.suptitle(title)
        
        if self._save:
            filename = f"scaleN_t_{time}"
            plt.savefig(self._save_root+filename+'.pdf')

        if self._show:
            plt.show()
    
    def N(self, g=None, g_list=None, N_list=None, time=1.0, xmin=None, xmax=None, ymin=None, ymax=None):
        if g is None:
            self._N_all(g_list=g_list, N_list=N_list, time=time, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        else:
            self._N_fixed_g(g=g, N_list=N_list, time=time, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    def _g_fixed_N(self, N, g_list=None, time=1.0, xmin=None, xmax=None, ymin=None, ymax=1.0):
        if g_list is None:
            g_list = self._g_list
        
        color = self._color_picker(N_lines=2)
        marker = self._marker_picker(N_lines=2)
        fig, axis = self._get_figure(N_figs=1)
        _wrapped_figure = FigureWrapper(fig)
        
        #nodes = self._nodes_picker(hidden_size, N)
        
        self._N = N
        x = g_list

        for i, data in enumerate(['Training', 'Testing']):
            y = np.empty(shape=len(x))
            y_err = np.empty(shape=len(x))
            for j, self._g in enumerate(g_list):
                self.load()
                #h = self._hidden_value_to_index(nodes)
                n = self._time_value_to_index(time)
                
                accuracy, avg, std, mse, mae = self._data_picker(data=data)
                #y[j] = accuracy[h,n]
                y[j] = avg[n]
                y_err[j] = std[n]
        
            axis[0].plot(x, y, linestyle='None', clip_on=self._clip_on, color=color[i], marker=marker[i], label=data, zorder=10)
            axis[0].plot(x, y, '-', clip_on=self._clip_on, color=color[i], zorder=10)
            axis[0].fill_between(x, y-y_err, y+y_err, alpha=0.5, antialiased=True, color=color[i], linewidth=0.0)
        
        xmin, xmax, ymin, ymax = self._axis_limits(x=x, y=y, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        
        if self._model == 'ising':
            axis[0].set_xlabel(r'$g / J$')
        elif self._model == 'xyz':
            axis[0].set_xlabel(r'$\Delta / J$')
        else:
            raise ValueError("Unrecognized model !")
        axis[0].set_ylabel('Acc.')
        axis[0].set_ylim(ymin=ymin, ymax=ymax)
        axis[0].set_xlim(xmin=xmin, xmax=xmax)
        axis[0].legend(frameon=False)
        
        if self._model == 'ising':
            title = f'{self._model}     N = {N}     $\\alpha$ = {self._alpha}     t = {time}'
        elif self._model == 'xyz':
            title = f'{self._model}     N = {N}     t = {time}'
        else:
            raise ValueError("Unrecognized model !")
        fig.suptitle(title)
        
        if self._save:
            filename = f"g_all_N_{self._N}_t_{time}"
            plt.savefig(self._save_root+filename+'.pdf')

        if self._show:
            plt.show()
    
    def _g_all_N(self, g_list=None, N_list=None, hidden_size=-1, time=1.0, xmin=None, xmax=None, ymin=None, ymax=1.0):
        if g_list is None:
            g_list = self._g_list
        if N_list is None:
            N_list = self._N_list
        
        color = self._color_picker(N_lines=len(N_list))
        marker = self._marker_picker(N_lines=len(N_list))
        fig, axis = self._get_figure(N_figs=2)
        _wrapped_figure = FigureWrapper(fig)
        
        x = g_list

        for i, data in enumerate(['Training', 'Testing']):
            for k, self._N in enumerate(N_list):
                y = np.empty(shape=len(x))
                y_err = np.empty(shape=len(x))
                for j, self._g in enumerate(g_list):
                    self.load()
                    n = self._time_value_to_index(time)
                    
                    accuracy, avg, std, mse, mae = self._data_picker(data=data)
                    y[j] = avg[n]
                    y_err[j] = std[n]

                axis[i].plot(x, y, linestyle='None', clip_on=self._clip_on, label=f'N = {self._N}', color=color[k], marker=marker[k], zorder=10)
                axis[i].plot(x, y, '-', clip_on=self._clip_on, color=color[k], zorder=10)
                axis[i].fill_between(x, y-y_err, y+y_err, alpha=0.5, antialiased=True, color=color[k], linewidth=0.0)
        
        xmin, xmax, ymin, ymax = self._axis_limits(x=x, y=y, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        
        axis[0].set_ylabel('Training Acc.')
        axis[0].set_ylim(ymin=ymin, ymax=ymax)
        axis[0].set_xlim(xmin=xmin, xmax=xmax)
        axis[0].legend(frameon=False)
        
        axis[1].set_ylabel('Testing Acc.')
        #axis[1].set_ylim(ymin=0.7, ymax=1.0)
        axis[1].set_ylim(ymin=ymin, ymax=ymax)
        axis[1].set_xlim(xmin=xmin, xmax=xmax)
        if self._model == 'ising':
            axis[1].set_xlabel(r'$g / J$')
        elif self._model == 'xyz':
            axis[1].set_xlabel(r'$\Delta / J$')
        else:
            raise ValueError("Unrecognized model !")

        if self._model == 'ising':
            title = f'{self._model}     $\\alpha$ = {self._alpha}     t = {time}'
        elif self._model == 'xyz':
            title = f'{self._model}     t = {time}'
        else:
            raise ValueError("Unrecognized model !")
        fig.suptitle(title)
        
        if self._save:
            filename = f"g_all_t_{time}"
            plt.savefig(self._save_root+filename+'.pdf')

        if self._show:
            plt.show()
    
    def g(self, N=None, g_list=None, N_list=None, time=1.0, xmin=None, xmax=None, ymin=None, ymax=1.0):
        if N is None:
            self._g_all_N(g_list=g_list, N_list=N_list, time=time, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        else:
            self._g_fixed_N(N=N, g_list=g_list, time=time, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)