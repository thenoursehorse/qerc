import numpy as np
from copy import deepcopy
from .nn import NeuralNetwork, AdaDelta, Adam, NAG, SGD, weight_initializer
        
class Perceptron(NeuralNetwork):
    def __init__(self, output_size, N_epochs=100, M=32, eta=0.001, beta_1=0.9, beta_2=0.999, eps=1e-7, optimizer='adam', initializer='xavier', shuffle=True, **kwargs):
        self._output_size = output_size
        self._N_epochs = N_epochs
        # Minibatch number. M = 1 is stochastic gradient descent (SGD), M = -1 batch gradient descent (BGD)
        self._M = M
        self._eta = eta
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._eps = eps
        self._optimizer = optimizer
        self._shuffle = shuffle
        self._initializer = initializer
        super().__init__(**kwargs)

        assert (self._activation == 'softmax'), 'Can only use softmax activation function for perceptron !'

        # Initialize perceptron weights
        self._weight, self._bias = weight_initializer(input_size=self._input_size, output_size=self._output_size, initializer=initializer)

        # Initialize optimizer
        if self._optimizer == 'adam':
            # good defaults are eta=0.001, beta_1=0.9, beta_2=0.999, eps=1e-7
            self._optimizer_w = Adam(eta=self._eta, beta_1=self._beta_1, beta_2=self._beta_2, eps=self._eps)
            self._optimizer_b = Adam(eta=self._eta, beta_1=self._beta_1, beta_2=self._beta_2, eps=self._eps)
        elif self._optimizer == 'adadelta':
            # good defaults are eta=0.01, rho=0.99, eps=1e-6
            self._optimizer_w = AdaDelta(eta=self._eta, beta_1=self._beta_1, beta_2=self._beta_2, eps=self._eps)
            self._optimizer_b = AdaDelta(eta=self._eta, beta_1=self._beta_1, beta_2=self._beta_2, eps=self._eps)
        elif self._optimizer == 'nag':
            # good defaults are eta=0.01, rho=0.9
            self._optimizer_w = NAG(eta=self._eta, gamma=self._beta_1)
            self._optimizer_b = NAG(eta=self._eta, gamma=self._beta_1)
        elif self._optimizer == 'sgd':
            self._optimizer_w = SGD(eta=self._eta)
            self._optimizer_b = SGD(eta=self._eta)
        else:
            raise ValueError(f'unknown activation optimizer {self._optimizer}.')

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
        return (x.T @ (y_pred - y)).T / M

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

        y_train_pred = np.empty(y_train.shape)
        y_test_pred = np.empty(y_test.shape)
        for i in range(self._N_epochs):
            
            # Record data at end of epoch
            y_train_pred = self.predict(x_train)
            self._accuracy_train[i], self._mse_train[i], self._mae_train[i] = self.evaluate(y_pred=y_train_pred, y=y_train)
            y_test_pred = self.predict(x_test)
            self._accuracy_test[i], self._mse_test[i], self._mae_test[i] = self.evaluate(y_pred=y_test_pred, y=y_test)

            # Shuffle data
            if self._shuffle:
                #idx = rng.permutation(N_samples)
                #idx = rng.integers(low=0, high=N_samples, size=M, endpoint=False)
                idx_rng = rng.integers(low=0, high=N_samples, size=N_samples, endpoint=False)

            for n in range(N_batch):
                # Determine the indices of the current batch (allows wrap around)
                idx = np.array(range(n*M, (n+1)*M)) % N_samples

                if self._shuffle:
                    idx = idx_rng[idx]
                
                x_train_copy = x_train[idx]
                y_train_copy = y_train[idx]
            
                # Get predictions for this batch
                y_train_pred_copy = self.activation_function(x_train_copy)

                # Get gradient
                w_deriv = self.weight_deriv(x=x_train_copy, y_pred=y_train_pred_copy, y=y_train_copy, M=M)
                b_deriv = self.bias_deriv(y_pred=y_train_pred_copy, y=y_train_copy, M=M)

                # Update weight and bias
                self._weight = self._optimizer_w.apply_gradients(vars=self._weight, dvars=w_deriv)
                self._bias = self._optimizer_b.apply_gradients(vars=self._bias, dvars=b_deriv)

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