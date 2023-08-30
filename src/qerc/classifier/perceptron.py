import numpy as np
from copy import deepcopy
from qerc.classifier.nn import NeuralNetwork, AdaDelta, Adam, NAG, SGD, weight_initializer, Annealing
        
class Perceptron(NeuralNetwork):
    '''
    Single-layer perceptron.

    Args:
        input_size : Size of input nodes.

        output_size : Size of output nodes.

        N_epochs : (Default 100) Number of iterations through the entire 
            training set.

        alpha : (Default 0.001) Learning rate hyperparameter.

        beta_1 : (Default 0.9) First moment derivitive mixing parameter. This 
            is the momentum hyperparameter.

        beta_2 : (Default 0.999) Second momentum derivitive mixing parameter. 
        
        eps : (Default 1e-7) Hyperparameter to prevent divide by zeros.

        optimizer : (Default 'adam') Optimization algorithm. Options are 
            'adam', 'adadelta', 'nag' Nesterov accelerated gradient, 
            'sgd' stochastic gradient descent.

        initializer : (Default 'xavier') Initialization scheme of weights for 
            nodes. Options are 'xavier', 'he', and 'zeros'.

        shuffle : (Default True) Whether to shuffle the training set at the 
            start of each epoch.
    '''
    def __init__(self, output_size, N_epochs=100, M=32, alpha=0.001, beta_1=0.9, beta_2=0.999, eps=1e-7, optimizer='adam', initializer='xavier', shuffle=True, **kwargs):
        self._output_size = output_size
        self._N_epochs = N_epochs
        # Minibatch number. M = 1 is stochastic gradient descent (SGD), M = -1 batch gradient descent (BGD)
        self._M = M
        self._alpha = alpha
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
            self._optimizer_w = Adam(alpha=self._alpha, beta_1=self._beta_1, beta_2=self._beta_2, eps=self._eps)
            self._optimizer_b = Adam(alpha=self._alpha, beta_1=self._beta_1, beta_2=self._beta_2, eps=self._eps)
        elif self._optimizer == 'adadelta':
            self._optimizer_w = AdaDelta(alpha=self._alpha, beta_1=self._beta_1, beta_2=self._beta_2, eps=self._eps)
            self._optimizer_b = AdaDelta(alpha=self._alpha, beta_1=self._beta_1, beta_2=self._beta_2, eps=self._eps)
        elif self._optimizer == 'nag':
            self._optimizer_w = NAG(alpha=self._alpha, gamma=self._beta_1)
            self._optimizer_b = NAG(alpha=self._alpha, gamma=self._beta_1)
        elif self._optimizer == 'sgd':
            self._optimizer_w = SGD(alpha=self._alpha)
            self._optimizer_b = SGD(alpha=self._alpha)
        else:
            raise ValueError(f'unknown activation optimizer {self._optimizer}.')

        # To hold data
        self._accuracy_train = np.empty(shape=self._N_epochs+1)
        self._accuracy_test = np.empty(shape=self._N_epochs+1)
        self._mse_train = np.empty(shape=self._N_epochs+1)
        self._mse_test = np.empty(shape=self._N_epochs+1)
        self._mae_train = np.empty(shape=self._N_epochs+1)
        self._mae_test = np.empty(shape=self._N_epochs+1)
        self._cross_entropy_train = np.empty(shape=self._N_epochs+1)
        self._cross_entropy_test = np.empty(shape=self._N_epochs+1)
        
        # FIXME not really implemented properly yet (and likely won't work well for such a simple network)
        #self.annealer = Annealing(N_epochs=self._N_epochs)
        #self.annealer = Annealing(N_epochs=self._N_epochs, T_mult=1, T0_mult=20)
        self.annealer = Annealing(N_epochs=self._N_epochs, anneal_type='flat')
           
    # [deriv]_ij = [X.T (Y_pred - Y)]_ij / N_samples
    def weight_deriv(self, x, y_pred, y, M):
        '''
        Analytic derivitive of cross entropy w.r.t. the weight.

        Args:
            x : Input array of some sample (first index labels sample).

            y_pred : Perceptrons guess at classifying (first index labels 
                sample).

            y : Correct classification (first index labels sample).

            M : Batch size (number of samples in this batch).

        Returns:
            Derivative dL/dw.
        '''
        return (x.T @ (y_pred - y)).T / M

    # [deriv]_j = \sum_k [Y_pred - Y]_j / N_samples
    def bias_deriv(self, y_pred, y, M):
        '''
        Analytic derivitive of cross entropy w.r.t. the bias.
        
        Args:
            y_pred : Perceptrons guess at classifying (first index labels 
                sample).

            y : Correct classification (first index labels sample).

            M : Batch size (number of samples in this batch).

        Returns:
            Derivative dL/db.
        '''
        return np.sum(y_pred - y, axis=0) / M

    def train(self, x_train, x_test, y_train, y_test):
        '''
        Trains the perceptron.

        Args:
            x_train : The input training data.

            x_test : The input testing/validation data.

            y_train : The required output of the perceptron of the training
                data. Should be onehot vectors.

            y_test : The required output of the perceptron of the 
                testing/validation data. Should be onehot vectors.

        Sets:
        At the beginning of each epoch:
            accuracy_train

            accuracy_test

            mse_train : mean-squared error

            mse_test 

            mae_train : mean absolute error

            mae_test

            cross_entropy_train : loss function for perceptron (cross entropy).

            cross_entropy_test

        Returns:
            y_train_pred : The predicted output of the perceptron on the 
                training data.

            y_test_pred : The predicted output of the perceptron on the
                testing data.
        '''
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
            #self._accuracy_train[i], self._mse_train[i], self._mae_train[i] = self.evaluate(y_pred=y_train_pred, y=y_train)
            self._accuracy_train[i], self._mse_train[i], self._mae_train[i], self._cross_entropy_train[i] = self.evaluate(y_pred=y_train_pred, y=y_train, calc_cross_entropy=True)
            y_test_pred = self.predict(x_test)
            #self._accuracy_test[i], self._mse_test[i], self._mae_test[i] = self.evaluate(y_pred=y_test_pred, y=y_test)
            self._accuracy_test[i], self._mse_test[i], self._mae_test[i], self._cross_entropy_test[i] = self.evaluate(y_pred=y_test_pred, y=y_test, calc_cross_entropy=True)

            # Scheduler
            eta = self.annealer.get_eta()

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
                self._weight = self._optimizer_w.apply_gradients(vars=self._weight, dvars=w_deriv, eta=eta)
                self._bias = self._optimizer_b.apply_gradients(vars=self._bias, dvars=b_deriv, eta=eta)

        # Record data of final iteration    
        y_train_pred = self.predict(x_train)
        #self._accuracy_train[-1], self._mse_train[-1], self._mae_train[-1] = self.evaluate(y_pred=y_train_pred, y=y_train)
        self._accuracy_train[-1], self._mse_train[-1], self._mae_train[-1], self._cross_entropy_train[-1] = self.evaluate(y_pred=y_train_pred, y=y_train, calc_cross_entropy=True)
        y_test_pred = self.predict(x_test)
        #self._accuracy_test[-1], self._mse_test[-1], self._mae_test[-1] = self.evaluate(y_pred=y_test_pred, y=y_test)
        self._accuracy_test[-1], self._mse_test[-1], self._mae_test[-1], self._cross_entropy_test[-1] = self.evaluate(y_pred=y_test_pred, y=y_test, calc_cross_entropy=True)
        return y_train_pred, y_test_pred
    
    def predict(self, x):
        '''
        Predict the classification of a single sample as a onehot vector. 
        x is an array that encodes information about the sample.
        '''
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
        '''
        Mean-squared error.
        '''
        return self._mse_train

    @property 
    def mse_test(self):
        '''
        Mean-squared error.
        '''
        return self._mse_test
    
    @property
    def mae_train(self):
        '''
        Mean-absolute error.
        '''
        return self._mae_train

    @property 
    def mae_test(self):
        '''
        Mean-absolute error.
        '''
        return self._mae_test
    
    @property
    def cross_entropy_train(self):
        '''
        Loss function (cross entropy).
        '''
        return self._cross_entropy_train

    @property 
    def cross_entropy_test(self):
        '''
        Loss function (cross entropy).
        '''
        return self._cross_entropy_test