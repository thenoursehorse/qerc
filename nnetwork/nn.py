
import numpy as np

class NeuralNetwork(object):
    def __init__(self, input_size, activation='softmax'):
        self._input_size = input_size
        self._activation = activation

    def activation_function(self, x):
        if self._activation == 'softmax':
            u = x @ self._weight.T + self._bias
            upper = np.exp(u)
            lower = np.sum(upper, axis=-1)
            out = np.empty(shape=upper.shape)
            for i in range(out.shape[0]):
                out[i,:] = upper[i,:] / lower[i]
            return out
        
        elif self._activation == 'sigmoid':
            u = x @ self._weight.T + self._bias
            return 1.0 / (1.0 + np.exp(-1.0 * u))
        
        elif self._activation == 'hyperbolic':
            u = x @ self._weight.T + self._bias
            a = np.exp(-u)
            return (1.0 - a) / (1.0 + a)

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
            return x

        else:
            raise ValueError(f'unknown activation function {self.activation}.')
    
    # Cross entropy
    # Indexed as # samples, output_size
    def cross_entropy(self, y_pred, y):
        return np.sum( - np.sum(y * np.log(y_pred), axis=1) ) / y.shape[0]

    def evaluate(self, y_pred, y, x_entropy=False):
        # Majority vote for 1 hot vectors
        y_pred_trunc = np.argmax(y_pred, axis=-1)
        y_trunc = np.argmax(y, axis=-1)

        accuracy = np.sum(y_pred_trunc == y_trunc) / y.shape[0]
        mse = np.mean((y_pred - y)**2)
        mae = np.mean(np.abs(y_pred - y))
        
        if x_entropy:
            x_entropy = self.cross_entropy(y_pred, y)
            return accuracy, mse, mae, x_entropy
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
        
    def standardize(self, x_train, x_test):
        from sklearn.preprocessing import StandardScaler
        std_scalar = StandardScaler()
        
        # Standardize output of training set
        std_scalar.fit(x_train)
        x_train_std = std_scalar.transform(x_train)
        
        # Apply to test
        x_test_std = std_scalar.transform(x_test)
        
        return x_train_std, x_test_std
            
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