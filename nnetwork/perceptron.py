import numpy as np
from copy import deepcopy
from .nn import NeuralNetwork
        
class Perceptron(NeuralNetwork):
    def __init__(self, output_size, N_epochs=100, M=32, eta=0.01, gamma=0.9, rho=0.99, eps=1e-6, shuffle=True, initialize_random=True, **kwargs):
        self._output_size = output_size
        self._N_epochs = N_epochs
        # Minibatch number. M = 1 is stochastic gradient descent (SGD), M = -1 batch gradient descent (BGD)
        self._M = M
        self._eta = eta
        self._gamma = gamma
        self._rho = rho
        self._eps = eps
        self._shuffle = shuffle
        self._initialize_random = initialize_random
        super().__init__(**kwargs)

        assert (self._activation == 'softmax'), 'Can only use softmax activation function for perceptron !'

        if (np.abs(self._rho) > 0.01) and (np.abs(self._gamma) > 0.01):
            print("WARNING: Doing momentum and AdaDelta at the same time can be risky. You should know what you're doing !")
            print("For AdaDelta: gamma = 0")
            print("For momentum: rho = 0")

        if self._initialize_random:
            # Randomize weights
            rng = np.random.default_rng()
            self._rand = rng.normal
            self._weight = self._rand(0, 1, (self._output_size, self._input_size)) * np.sqrt(1.0/(self._input_size)) # He initialization
            #self._weight = self._rand(0, 1, (self._output_size, self._input_size)) * np.sqrt(2.0/(self._input_size + self._output_size)) # Xavier initialization
            self._bias = np.zeros(shape=(self._output_size))
        
            #self._rand = rng.uniform
            #self._weight = self._rand(-1, 1, (self._output_size, self._input_size))
            #self._bias = self._rand(-1, 1, (self._output_size))
        else:
            # Initialize weights to zero
            self._weight = np.zeros(shape=(self._output_size, self._input_size))
            self._bias = np.zeros(shape=(self._output_size))

        # To hold data
        self._accuracy_train = np.empty(shape=self._N_epochs+1)
        self._accuracy_test = np.empty(shape=self._N_epochs+1)
        self._mse_train = np.empty(shape=self._N_epochs+1)
        self._mse_test = np.empty(shape=self._N_epochs+1)
        self._mae_train = np.empty(shape=self._N_epochs+1)
        self._mae_test = np.empty(shape=self._N_epochs+1)
        self._losses_train = np.empty(shape=self._N_epochs+1)
        self._losses_test = np.empty(shape=self._N_epochs+1)
           
    # [deriv]_ij = [X.T (Y_pred - Y)]_ij / N_samples
    def weight_deriv(self, x, y_pred, y, M):
        return x.T @ (y_pred - y) / M

    # [deriv]_j = \sum_k [Y_pred - Y]_j / N_samples
    def bias_deriv(self, y_pred, y, M):
        return np.sum(y_pred - y, axis=0) / M

    def train(self, x_train, x_test, y_train, y_test):
        rng = np.random.default_rng()

        N_samples = y_train.shape[0]
        if self._M < 0:
            M = N_samples
        else:
            M = self._M
        N_batch = int(N_samples / M)

        # AdaDelta is from arXiv:1212.5701v1
        # To store AdaDelta learning rate
        # if rho = 0 then AdaDelta is turned off
        Eg_w = 0.0
        Eg_b = 0.0
        Ed_w = self._eta**2
        Ed_b = self._eta**2

        y_train_pred = np.empty(y_train.shape)
        y_test_pred = np.empty(y_test.shape)
        for i in range(self._N_epochs):
            
            # Record data at end of epoch
            y_train_pred = self.predict(x_train)
            self._accuracy_train[i], self._mse_train[i], self._mae_train[i] = self.evaluate(y_pred=y_train_pred, y=y_train)
            y_test_pred = self.predict(x_test)
            self._accuracy_test[i], self._mse_test[i], self._mae_test[i] = self.evaluate(y_pred=y_test_pred, y=y_test)

            for n in range(N_batch):
            
                # Shuffle data
                if self._shuffle:
                    #idx = rng.permutation(N_samples)
                    idx = rng.integers(low=0, high=N_samples, size=M, endpoint=False)
                    x_train_copy = x_train[idx]
                    y_train_copy = y_train[idx]
                else:
                    # Determine the indices of the current batch (allows wrap around)
                    idx = np.array(range(n*M, (n+1)*M)) % N_samples
                    x_train_copy = x_train[idx]
                    y_train_copy = y_train[idx]
            
                # Get predictions for this batch
                y_train_pred_copy = self.activation_function(x_train_copy)

                # Get gradient
                w_deriv = self.weight_deriv(x=x_train_copy, y_pred=y_train_pred_copy, y=y_train_copy, M=M)
                b_deriv = self.bias_deriv(y_pred=y_train_pred_copy, y=y_train_copy, M=M)
            
                # Accumulate gradients for AdaDelta
                Eg_w = self._rho * Eg_w + (1.0 - self._rho) * w_deriv**2
                Eg_b = self._rho * Eg_b + (1.0 - self._rho) * b_deriv**2
            
                # Update direction adapted by AdaDelta
                #d_w = (self._eta / np.sqrt(Eg_w + self._eps)).T * w_deriv.T
                #d_b = (self._eta / np.sqrt(Eg_b + self._eps)) * b_deriv
                d_w = (np.sqrt(Ed_w + self._eps).T / np.sqrt(Eg_w + self._eps)).T * w_deriv.T
                d_b = (np.sqrt(Ed_b + self._eps) / np.sqrt(Eg_b + self._eps)) * b_deriv

                # Add momentum (default to 0 because adadelta is the default step size)
                d_w += self._gamma * np.sqrt(Ed_w + self._eps)
                d_b += self._gamma * np.sqrt(Ed_b + self._eps)
            
                # Accumulate updates
                Ed_w = self._rho * Ed_w + (1.0 - self._rho) * d_w**2
                Ed_b = self._rho * Ed_b + (1.0 - self._rho) * d_b**2
            
                # Update weight and bias
                self._weight -= d_w
                self._bias -= d_b
            
        # Record data of final iteration    
        y_train_pred = self.predict(x_train)
        self._accuracy_train[-1], self._mse_train[-1], self._mae_train[-1] = self.evaluate(y_pred=y_train_pred, y=y_train)
        y_test_pred = self.predict(x_test)
        self._accuracy_test[-1], self._mse_test[-1], self._mae_test[-1] = self.evaluate(y_pred=y_test_pred, y=y_test)
        return y_train_pred, y_test_pred
    
    def predict(self, x):
        return self.activation_function(x)
    
    @property
    def output_size(self):
        return self._output_size

    @property
    def accuracy_train(self):
        return self._accuracy_train

    @property 
    def accuracy_test(self):
        return self._accuracy_test
    
    @property
    def mse_train(self):
        return self._mse_train

    @property 
    def mse_test(self):
        return self._mse_test
    
    @property
    def mae_train(self):
        return self._mae_train

    @property 
    def mae_test(self):
        return self._mae_test
    
    @property
    def losses_train(self):
        return self._losses_train

    @property 
    def losses_test(self):
        return self._losses_test