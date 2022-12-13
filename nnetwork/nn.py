import numpy as np
from copy import deepcopy

class Annealing(object):
    def __init__(self, N_epochs, anneal_type='cosine', T_mult=2, T0_mult=0.05, eta_min=0, eta_max=1):
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.eta_max = eta_max

        self.T_cur = 0
        self.T = T0_mult * N_epochs # 1-10% of total budget is recommendation in cosine warm restart 
        self.resets = 0 # i in the cosine warm restart
        
        self.eta = 0

        if anneal_type == 'flat':
            self.get_eta = self._get_flat
        elif anneal_type == 'cosine':
            self.get_eta = self._get_cosine
        else:
            raise ValueError("unrecognized annealing type !")

    # cosine warm restart annealing is from Appendix B in arXiv:1711.05101v3
    def _get_cosine(self):
        self.eta = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1.0 + np.cos(np.pi * self.T_cur / self.T))
            
        self.T_cur += 1

        # Check restart
        if self.T_cur >= self.T:
            self.T_cur = 0
            self.T *= self.T_mult
            self.resets += 1
            
        return self.eta
        
    def _get_flat(self):
        self.eta = 1.0
        return self.eta

# Adam is from arXiv:1412.6980v9
class Adam(object):
    def __init__(self, alpha=0.001, beta_1=0.9, beta_2=0.999, eps=1e-7, use_nadam=False, use_amsgrad=True):
        self.beta_1 = beta_1 # RMSProp is if beta_1 = 0
        self.beta_2 = beta_2
        self.alpha = alpha
        self.eps = eps
        
        # for using Nesterov momentum (in our data it makes very little difference)
        self.use_nadam = use_nadam
        
        # For using the AMSGrad algorithm from https://openreview.net/forum?id=ryQu7f-RZ
        self.use_amsgrad = use_amsgrad
        
        self.t = 1
        self.m = 0.0
        self.v = 0.0
        self.v_old = None

    def apply_gradients(self, vars, dvars, eta=1.0):
        
        if self.use_amsgrad:
            # Store old second moment
            self.v_old = deepcopy(self.v)

        # First and second moment estimate
        self.m = self.beta_1*self.m + (1.0 - self.beta_1) * dvars
        self.v = self.beta_2*self.v + (1.0 - self.beta_2) * dvars**2
        
        # Bias correction to moments
        m_bc = self.m / (1.0 - self.beta_1**self.t)
        v_bc = self.v / (1.0 - self.beta_2**self.t)

        if self.use_amsgrad:
            # Fix for the exponential moving average
            v_bc = np.maximum(v_bc, self.v_old)
        
        # Update direction to vars
        if self.use_nadam:
            updates = (self.alpha / (np.sqrt(v_bc) + self.eps)) * ( self.beta_1 * m_bc + (1.0 - self.beta_1)*dvars / (1.0 - self.beta_1**self.t) )
        else:
            updates = self.alpha * m_bc / (np.sqrt(v_bc) + self.eps)

        # Here is where I would add weight decay for regularization, but it makes little difference
        # for such shallow networks
        # This part of the algorithm is AdamW

        # Apply scheduler for learning
        updates *= eta
        
        self.t += 1
        vars -= updates
        return vars

# AdaDelta is from arXiv:1212.5701v1
# This has been adapted to be like Adam.
class AdaDelta(object):
    def __init__(self, alpha=0.001, beta_1=0.9, beta_2=0.999, eps=1e-7, use_amsgrad=True):
        self.alpha = alpha
        self.beta_1 = beta_1 # beta = 0 then becomes default Adadelta
        self.beta_2 = beta_2 # beta = 0 then becomes SGD/momentum
        self.eps = eps
        
        # For using the AMSGrad algorithm from https://openreview.net/forum?id=ryQu7f-RZ
        self.use_amsgrad = use_amsgrad

        self.t = 1
        self.m = 0.0
        self.v = 0
        self.v_old = None
        
        # To store rolling average of update weight
        self.Ed = self.alpha**2
        self.Ed_old = None

    def apply_gradients(self, vars, dvars, eta=1.0):
        
        if self.use_amsgrad:
            # Store old second moment
            self.v_old = deepcopy(self.v)

        # First and second moment estimate
        self.m = self.beta_1*self.m + (1.0 -self.beta_1) * dvars
        self.v = self.beta_2 * self.v + (1.0 - self.beta_2) * dvars**2
        
        if self.use_amsgrad:
            # Fix for the exponential moving average
            self.v = np.maximum(self.v, self.v_old)
            
        # Update direction adapted by AdaDelta (major difference to adam, and has no bias correction)
        updates = self.m * np.sqrt(self.Ed + self.eps) / np.sqrt(self.v + self.eps)

        if self.use_amsgrad:
            # Adadelta also requires taking the max of this
            self.Ed_old = deepcopy(self.Ed)

        # Accumulate updates for rolling avg
        self.Ed = self.beta_2 * self.Ed + (1.0 - self.beta_2) * updates**2

        if self.use_amsgrad:
            self.Ed = np.maximum(self.Ed, self.Ed_old)
        
        # Apply scheduler for learning
        updates *= eta

        self.t += 1
        vars -= updates
        return vars

# Nesterov accelerated gradient
# Nesterov, Y. (1983). A method for unconstrained convex minimization problem with the rate of convergence o(1/k2).
class NAG(object):
    def __init__(self, alpha=0.01, gamma=0.9, no_look_ahead=False):
        self.alpha = alpha
        self.gamma = gamma
        self.no_look_ahead = no_look_ahead
        self.m = 0
        self.t = 1

    def apply_gradients(self, vars, dvars, eta=1.0):
        # Get standard momentum update rule
        self.m = self.gamma * self.m + self.alpha*dvars
        
        # Apply Nesterov look ahead implementation
        if no_look_ahead:
            updates = self.m
        else:
            updates = self.gamma * self.m + self.alpha*dvars
        
        # Apply scheduler for learning
        updates *= eta
        
        self.t += 1
        vars -= updates
        return vars

class SGD(object):
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.t = 1

    def apply_gradients(self, vars, dvars, eta=1.0):
        updates = self.alpha * dvars
        
        # Apply scheduler for learning
        updates *= eta
        
        self.t += 1
        vars -= updates
        return vars

def weight_initializer(input_size, output_size, initializer='xavier'):
            
    rng = np.random.default_rng()
    normal = rng.normal
    uniform = rng.uniform

    # Weights usually get randomized to a value near zero
    if initializer == 'xavier':
        scale = np.sqrt(6.0/(input_size + output_size))
        weight = uniform(low=-scale, high=scale, size=(output_size, input_size))
    elif initializer == 'xavier2':
        weight = normal(loc=0, scale=1, size=(output_size, input_size)) * np.sqrt(2.0/(input_size + output_size))
    elif initializer == 'he':
        weight = normal(loc=0, scale=1, size=(output_size, input_size)) * np.sqrt(1.0/(input_size))
    elif initializer == 'zeros':
        weight = np.zeros(shape=(output_size,input_size))
    else:
        raise ValueError(f'unknown weight initializer {initialize_type}.')

    # Bias is always zero
    bias = np.zeros(shape=(output_size))
    
    return weight, bias
        
# For applying gradient noise from arXiv:1511.06807v1 (makes very little difference for shallow networks like ours)
def gradient_noise(dvars, t, alpha=0.01, gamma=0.55):
    # parameters from their eq. 1 alpha in (0.01, 0.3, 1), gamma=0.55
    rng = np.random.default_rng()
    rand = rng.normal
    variance = alpha / (1.0 + t)**gamma
    noise = rand(0,var,dvars.shape)
    dvars += noise
    return dvars

# Weight decay for L2 regularization from arXiv:1711.05101v3
def weight_decay(updates, vars, N_batch, N_samples, N_epochs):
    lambdaa = 0.25*0.001 # some initial guess from their Fig.1,2
    lambdaa *= np.sqrt(N_batch/(N_samples*N_epochs)) # their renormalized value from appendix
    updates += lambdaa * vars
    return updates

class NeuralNetwork(object):
    def __init__(self, input_size, activation='softmax', identity_bias=True):
        self._input_size = input_size
        self._activation = activation
        self._identity_bias = identity_bias

    def _log_sum_exp(self, x):
        A = x.max(axis=-1)
        return A + np.log(np.sum(np.exp(x - A[:,None]), axis=-1))

    def _sum_exp(self, x):
        e = np.exp(x - np.max(x, axis=-1)[:,None])
        return e / np.sum(e, axis=-1, keepdims=True)

    def activation_function(self, x):
        if self._activation == 'softmax':
            u = x @ self._weight.T + self._bias

            #lower = self._log_sum_exp(u)
            #u -= lower[:,None]
            #return np.exp(u)

            return self._sum_exp(u)

        elif self._activation == 'sigmoid':
            u = x @ self._weight.T + self._bias
            return 1.0 / (1.0 + np.exp(-1.0 * u))
        
        elif self._activation == 'hyperbolic':
            u = x @ self._weight.T + self._bias
            e = np.exp(-u)
            return (1.0 - e) / (1.0 + e)

        elif self._activation == 'hard':
            u = x @ self._weight.T + self._bias
            idx_gr = u > 0
            idx_ls = u <= 0
            u[ idx_gr ] = 0
            u[ idx_ls ] = 1
            return u

        elif self._activation == 'cos':
            u = x @ self._weight.T + self._bias
            return np.cos(u)

        elif self._activation == 'identity':
            # From PRL 127, 100502 (2021) adding a constant bias of 1 helps optimize the training
            if self._identity_bias:
                return np.hstack([x, np.ones((x.shape[0],1))])
            else:
                return x

        else:
            raise ValueError(f'unknown activation function {self.activation}.')
    
    # Cross entropy
    # Indexed as # samples, output_size
    def cross_entropy(self, y_pred, y):
        return np.sum( - np.sum(y * np.log(y_pred), axis=-1) ) / y.shape[0]
        #return np.sum(np.log( np.prod(np.exp(-y) * y_pred, axis=-1) )) / y.shape[0]

    def evaluate(self, y_pred, y, calc_cross_entropy=False):
        # Majority vote for 1 hot vectors
        y_pred_trunc = np.argmax(y_pred, axis=-1)
        y_trunc = np.argmax(y, axis=-1)
        
        accuracy = np.sum(y_pred_trunc == y_trunc) / y.shape[0]
        mse = np.mean((y_pred - y)**2)
        mae = np.mean(np.abs(y_pred - y))
        
        if calc_cross_entropy:
            cross_entropy = self.cross_entropy(y_pred=y_pred, y=y)
            return accuracy, mse, mae, cross_entropy
        return accuracy, mse, mae
        
    def onehot_y(self, y_train, y_test, M=10, binarizer=True):      
        # Use sklearn binarizer to make one hot vectors
        if binarizer:
            from sklearn.preprocessing import LabelBinarizer
            lb = LabelBinarizer()
            lb.fit(y_train)
            y_train = lb.transform(y_train).astype(int)
            y_test = lb.transform(y_test).astype(int)
            return y_train, y_test
        
        # Make one hot vectors of M categories
        else:
            N_samples_train = len(y_train)
            T_train = np.zeros(shape=(N_samples_train, M))
            x = [i for i in range(N_samples_train)]
            T_train[x,y_train] = 1
            
            N_samples_test = len(y_test)
            T_test = np.zeros(shape=(N_samples_test, M))
            x = [i for i in range(N_samples_test)]
            T_test[x,y_test] = 1
            
            return T_train, T_test
        
    def standardize(self, x_train, x_test, individual=True):
        if individual:
            # Standardize each sample to individually have mean 0 and std deviation 1
            x_train = ( x_train - np.mean(x_train, axis=-1, keepdims=True) ) / ( np.std(x_train, axis=-1, keepdims=True) + 1e-10 )
            x_test = ( x_test - np.mean(x_test, axis=-1, keepdims=True) ) / ( np.std(x_test, axis=-1, keepdims=True) + 1e-10 )

        else:
            # Standardize output of training set
            from sklearn.preprocessing import StandardScaler
            std_scalar = StandardScaler()
            std_scalar.fit(x_train)
            x_train = std_scalar.transform(x_train)
            x_test = std_scalar.transform(x_test)
        
        return x_train, x_test
            
    @property
    def input_size(self):
        return self._input_size

    @property
    def activation(self):
        return self._activation
    
    @property
    def weight(self):
        return self._weight
    
    @property
    def bias(self):
        return self._bias        