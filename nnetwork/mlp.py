import numpy as np
from .nn import NeuralNetwork

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

def xavier_init(shape):
    # Computes the xavier initialization values for a weight matrix
    in_dim, out_dim = shape
    xavier_lim = tf.sqrt(6.)/tf.sqrt(tf.cast(in_dim + out_dim, tf.float32))
    weight_vals = tf.random.uniform(shape=(in_dim, out_dim),
                                    minval=-xavier_lim, maxval=xavier_lim, seed=22)
    return weight_vals

def cross_entropy_loss(y_pred, y):
    # Compute cross entropy loss with a sparse operation
    sparse_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_pred)
    return tf.reduce_mean(sparse_ce)

def accuracy(y_pred, y):
    # Compute accuracy after extracting class predictions
    class_preds = tf.argmax(tf.nn.softmax(y_pred), axis=1)
    is_equal = tf.equal(y, class_preds)
    return tf.reduce_mean(tf.cast(is_equal, tf.float32))

def train_step(x_batch, y_batch, loss, acc, model, optimizer):
    # Update the model state given a batch of data
    with tf.GradientTape() as tape:
        y_pred = model(x_batch)
        batch_loss = loss(y_pred, y_batch)
    batch_acc = acc(y_pred, y_batch)
    grads = tape.gradient(batch_loss, model.variables)
    optimizer.apply_gradients(grads, model.variables)
    return batch_loss, batch_acc

def val_step(x_batch, y_batch, loss, acc, model):
    # Evaluate the model on given a batch of validation data
    y_pred = model(x_batch)
    batch_loss = loss(y_pred, y_batch)
    batch_acc = acc(y_pred, y_batch)
    return batch_loss, batch_acc

class MLPGenerator(tf.Module):
    def __init__(self, layers):
        self.layers = layers

    @tf.function
    def __call__(self, x, preds=False):
        # Execute the model's layers sequentially
        for layer in self.layers:
            x = layer(x)
        return x

class DenseLayer(tf.Module):
    def __init__(self, out_dim, weight_init=xavier_init, activation=tf.identity):
        # Initialize the dimensions and activation functions
        self.out_dim = out_dim
        self.weight_init = weight_init
        self.activation = activation
        self.built = False

    def __call__(self, x):
        if not self.built:
            # Infer the input dimension based on first call
            self.in_dim = x.shape[1]
            # Initialize the weights and biases using Xavier scheme
            self.w = tf.Variable(xavier_init(shape=(self.in_dim, self.out_dim)))
            self.b = tf.Variable(tf.zeros(shape=(self.out_dim,)))
            self.built = True
        # Compute the forward pass
        #z = tf.add(tf.matmul(x, self.w), self.b)
        z = tf.add(tf.matmul(tf.cast(x, tf.float32), self.w), self.b)
        return self.activation(z)

class Adam:
    def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, ep=1e-7):
        # Initialize optimizer parameters and variable slots
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.learning_rate = learning_rate
        self.ep = ep
        self.t = 1.
        self.v_dvar, self.s_dvar = [], []
        self.built = False

    def apply_gradients(self, grads, vars):
        # Initialize variables on the first call
        if not self.built:
            for var in vars:
                v = tf.Variable(tf.zeros(shape=var.shape))
                s = tf.Variable(tf.zeros(shape=var.shape))
                self.v_dvar.append(v)
                self.s_dvar.append(s)
                self.built = True
        # Update the model variables given their gradients
        for i, (d_var, var) in enumerate(zip(grads, vars)):
            self.v_dvar[i].assign(self.beta_1*self.v_dvar[i] + (1-self.beta_1)*d_var)
            self.s_dvar[i].assign(self.beta_2*self.s_dvar[i] + (1-self.beta_2)*tf.square(d_var))
            v_dvar_bc = self.v_dvar[i]/(1-(self.beta_1**self.t))
            s_dvar_bc = self.s_dvar[i]/(1-(self.beta_2**self.t))
            var.assign_sub(self.learning_rate*(v_dvar_bc/(tf.sqrt(s_dvar_bc) + self.ep)))
        self.t += 1.
        return

class MLP(NeuralNetwork):
    def __init__(self, output_size, N_epochs=10, M=128, eta=0.01, shuffle=True, **kwargs):
        self._output_size = output_size
        self._N_epochs = N_epochs
        # Minibatch number. M = 1 is stochastic gradient descent (SGD), M = -1 batch gradient descent (BGD)
        self._M = M
        self._eta = eta
        self._shuffle = shuffle
        super().__init__(**kwargs)

        #assert (self._activation == 'softmax'), 'Can only use softmax activation function for perceptron !'

        #hidden_layer_1_size = 700
        #hidden_layer_2_size = 500
        #output_size = 10
        #
        #mlp_model = MLPGenerator([
        #    DenseLayer(out_dim=hidden_layer_1_size, activation=tf.nn.relu),
        #    DenseLayer(out_dim=hidden_layer_2_size, activation=tf.nn.relu),
        #    DenseLayer(out_dim=output_size)])

        self._mlp_model = MLPGenerator([
            DenseLayer(out_dim=output_size)])

        self._loss = cross_entropy_loss
        self._accuracy = accuracy
        self._optimizer = Adam()

        # To hold data
        self._accuracy_train = np.empty(shape=self._N_epochs+1)
        self._accuracy_test = np.empty(shape=self._N_epochs+1)
        self._mse_train = np.empty(shape=self._N_epochs+1)
        self._mse_test = np.empty(shape=self._N_epochs+1)
        self._mae_train = np.empty(shape=self._N_epochs+1)
        self._mae_test = np.empty(shape=self._N_epochs+1)
        self._losses_train = np.empty(shape=self._N_epochs+1)
        self._losses_test = np.empty(shape=self._N_epochs+1)

    def train(self, x_train, x_test, y_train, y_test):
        # Onehot to labels
        if len(y_train.shape) == 2:
            y_train = np.argmax(y_train, axis=1)
        if len(y_test.shape) == 2:
            y_train = np.argmax(y_test, axis=1)

        N_samples = y_train.shape[0]
        if self._M < 0:
            M = N_samples
        else:
            M = self._M
        N_batch = int(N_samples / M)
        
        # Format training loop and begin training
        for i in range(self._N_epochs):
            
            ## Record data at end of epoch
            #y_train_pred[...] = self.predict(x_train)[...]
            #self._accuracy_train[i], self._mse_train[i], self._mae_train[i] = self.evaluate(y_pred=y_train_pred, y=y_train)
            #y_test_pred[...] = self.predict(x_test)[...]
            #self._accuracy_test[i], self._mse_test[i], self._mae_test[i] = self.evaluate(y_pred=y_test_pred, y=y_test)
            
            # Shuffle data
            if self._shuffle:
                idx_rng = rng.permutation(N_samples)
                x_train_copy = x_train[idx_rng]
                y_train_copy = y_train[idx_rng]
            else:
                x_train_copy = x_train
                y_train_copy = y_train
            
            batch_losses_train, batch_accs_train = [], []
            batch_losses_val, batch_accs_val = [], []
            for n in range(N_batch):
                # Determine the indices of the current batch (allows wrap around)
                idx = np.array(range(n*M, (n+1)*M)) % N_samples

                # Compute gradients and update the model's parameters
                batch_loss, batch_acc = train_step(x_train_copy[idx,:], y_train_copy[idx], self._loss, self._accuracy, self._mlp_model, self._optimizer)
                # Keep track of batch-level training performance
                batch_losses_train.append(batch_loss)
                batch_accs_train.append(batch_acc)

            # Iterate over the validation data
            #for x_batch, y_batch in val_data:
            #    batch_loss, batch_acc = val_step(x_batch, y_batch, self._loss, self._accuracy, self._mlp_model)
            #    batch_losses_val.append(batch_loss)
            #    batch_accs_val.append(batch_acc)

            # Keep track of epoch-level model performance
            loss_train, acc_train = tf.reduce_mean(batch_losses_train), tf.reduce_mean(batch_accs_train)
            val_loss, val_acc = tf.reduce_mean(batch_losses_val), tf.reduce_mean(batch_accs_val)

            self._losses_train[i] = loss_train
            self._accuracy_train[i] = acc_train
            
            #val_losses.append(val_loss)
            #val_accs.append(val_acc)
            print(f"Epoch: {i}")
            print(f"Training loss: {loss_train:.3f}, Training accuracy: {acc_train:.3f}")
            #print(f"Validation loss: {val_loss:.3f}, Validation accuracy: {val_acc:.3f}")
        
        ## Record data of final iteration    
        #y_train_pred[...] = self.predict(x_train)[...]
        #self._accuracy_train[-1], self._mse_train[-1], self._mae_train[-1] = self.evaluate(y_pred=y_train_pred, y=y_train)
        #y_test_pred[...] = self.predict(x_test)[...]
        #self._accuracy_test[-1], self._mse_test[-1], self._mae_test[-1] = self.evaluate(y_pred=y_test_pred, y=y_test)
        #return y_train_pred, y_test_pred
        return None, None
           
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