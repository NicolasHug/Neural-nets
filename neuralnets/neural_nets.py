import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder

from .utils import ACTIVATIONS

class NeuralNet(BaseEstimator):
    """A neural network model.

    Args:
        n_neurons (list):list of H + 2 integers indicating the number of
            neurons in each layer, including input and output layers. First
            value is the number of features of a training example. Last value
            is either 1 (2, classes, logistic loss) or C > 1 (C > 2 classes,
            cross entropy), where C is the number of classes. In-between, the H
            values indicate the number of neurons of each of the H hidden
            layer.
        activations (str or list of str): The activation functions to use for
            each of the H hidden layers. Allowed values are 'sigmoid', 'tanh',
            'relu' or 'linear' (i.e. no activation). If a str is given, then
            all activations are the same for each hidden layer. If a list of
            string is given, it must be of size H. Default is 'relu'. Note: the
            activation function of the last layer is automatically inferred
            from the value of n_neurons[-1]: if the output layer size is 1 then
            a sigmoid is used, else it's a softmax.
        learning_rate(float): The learning rate for gradient descent. Default
            is .005.
        n_epochs(int): The number of iteration of the gradient descent
            procedure, i.e. number of times the whole training set is gone
            through. Default is 200.
        batch_size(int): The batch size. If 0, the full trainset is used.
            Default is 64.
        dropout(float): Probability of keeping a neuron of the hidden layers in
            dropout. Default is 1, i.e. no dropout is applied.
        lambda_reg(float): The regularization constant. Default is 0, i.e. no
            regularization.
        init_strat(str): Initialization strategy for weights. Can be 'He' for
            'He' initialization, recommended for relu layers. Default is None,
            which reverts to a centered normal distribution * 0.1.
        solver(str): Solver to use: either 'sgd' or 'adam' for SGD or... adam
            ;). Default is 'sgd'.
        beta_1(float): Exponential decay rate for first moment estimate (only
            used if solver is 'adam'. Default is .9
        beta_2(float): Exponential decay rate for second moment estimate (only
            used if solver is 'adam'. Default is .999.
        seed(int): A random seed to use for the RNG at weights initialization.
            Default is None, i.e. no seeding is done.
        check_gradients(bool): Whether to check gradients at each iteration,
            for each parameter. It's done with np.isclose() with default
            tolerance values. Default is False.
        verbose(int): if not False or 0, will print the loss every 'verbose'
            epochs. Default is False.


    Note: The NeuralNet estimator is (roughly) compliant with scikit-learn API
    so the inputs X fit and predict is [n_entries, n_features], but internally
    we use X.T because it seems more convenient.
    """

    def __init__(self, n_neurons, activations='relu', learning_rate=.005,
                 n_epochs=200, batch_size=64, dropout=1, lambda_reg=0,
                 init_strat=None, solver='sgd', beta_1=.9, beta_2=.999,
                 seed=None, check_gradients=False, verbose=False):

        self.n_neurons = n_neurons
        self.n_layers = len(self.n_neurons)  # including input and output
        self.init_activations(activations)
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.lbd = lambda_reg
        self.gd_step = self.sgd if solver == 'sgd' else self.adam
        self.do_grad_check = check_gradients
        self.verbose = verbose
        self.losses = []

        if n_neurons[-1] == 1:  # 2 classes problem, 1 output neuron
            self.compute_loss = self.logistic_loss
        else:  # C > 2 classes problem, C output neurons
            self.compute_loss = self.cross_entropy_loss

        self.init_params(seed, init_strat)

        if solver != 'sgd':  # i.e. adam
            self.init_adam(beta_1, beta_2)

    def init_activations(self, activations):
        '''Initialize activations.'''

        if isinstance(activations, str):  # transform str into list of same str
            activations = [activations] * (self.n_layers - 2)
        self.activations = dict()
        self.activations_deriv = dict()

        for i, activation in enumerate(activations):
            try:
                act_fun, deriv_fun = ACTIVATIONS[activation]
                self.activations[i + 1] = act_fun
                self.activations_deriv[i + 1] = deriv_fun
            except KeyError:
                exit('Unsupported activation' + activation)

        # Last layer: either sigmoid or softmax
        act_fun, deriv_fun = (ACTIVATIONS['sigmoid'] if self.n_neurons[-1] == 1
                              else ACTIVATIONS['softmax'])
        self.activations[self.n_layers - 1] = act_fun
        self.activations_deriv[self.n_layers - 1] = deriv_fun

    def init_params(self, seed, init_strat):
        '''Initialize weights and biases.'''

        if seed is not None:
            np.random.seed(seed)
        self.W = dict()
        self.b = dict()
        for l in range(1, self.n_layers):
            if init_strat == 'He':
                self.W[l] = (np.random.randn(self.n_neurons[l],
                                             self.n_neurons[l - 1]) *
                             np.sqrt(2 / self.n_neurons[l - 1]))
            else:
                self.W[l] = np.random.randn(self.n_neurons[l],
                                             self.n_neurons[l - 1]) * .01

            self.b[l] = np.zeros((self.n_neurons[l], 1))

    def init_adam(self, beta_1, beta_2):
        '''Initialize adam parameters.'''
        layers = range(1, self.n_layers)
        self.mW = {l: np.zeros(self.W[l].shape) for l in layers}
        self.vW = {l: np.zeros(self.W[l].shape) for l in layers}
        self.mb = {l: np.zeros(self.b[l].shape) for l in layers}
        self.vb = {l: np.zeros(self.b[l].shape) for l in layers}
        self.epsilon = 10E-8
        self.adam_counter = 0  # number of times self.adam is called
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def fit(self, X, y):
        """ Fit model with input X[n_entries, n_features] and output y."""

        def convert_X_y(X, y):
            if self.n_neurons[-1] != 1:  # if C > 2
                enc = OneHotEncoder(sparse=False)
                y = enc.fit(y[:, np.newaxis]).transform(y[:, np.newaxis]).T
            else:
                y = y[np.newaxis, :]

            return X.T, y

        X, y = convert_X_y(X, y)

        for current_epoch in range(self.n_epochs):
            batches = self.get_batches(X, y)
            for X_b, y_b in batches:
                y_hat, cache = self.forward(X_b)
                dW, db = self.backward(X_b, y_b, cache)

                if self.do_grad_check:
                    self.check_gradients(X_b, y_b, dW, db)

                self.gd_step(dW, db)

            # Compute loss at the end of each epoch
            y_hat, _ = self.forward(X)
            loss = self.compute_loss(y_hat, y)
            self.losses.append(loss)
            if self.verbose and current_epoch % self.verbose == 0:
                print('Epoch {0:5d}, loss= {1:1.3f}'.format(current_epoch, loss))

        return self

    def sgd(self, dW, db):
        '''SGD step.'''
        for l in range(1, self.n_layers):
            self.W[l] -= self.lr * dW[l]
            self.b[l] -= self.lr * db[l]

    def adam(self, dW, db):
        '''Adam step.'''
        self.adam_counter += 1
        for l in range(1, self.n_layers):
            self.mW[l] = self.beta_1 * self.mW[l] + (1 - self.beta_1) * dW[l]
            self.vW[l] = self.beta_2 * self.vW[l] + (1 - self.beta_2) * dW[l]**2
            m_unbiased = self.mW[l] / (1 - self.beta_1**self.adam_counter)
            v_unbiased = self.vW[l] / (1 - self.beta_2**self.adam_counter)
            self.W[l] -= self.lr * m_unbiased / (np.sqrt(v_unbiased) + self.epsilon)

            self.mb[l] = self.beta_1 * self.mb[l] + (1 - self.beta_1) * db[l]
            self.vb[l] = self.beta_2 * self.vb[l] + (1 - self.beta_2) * db[l]**2
            m_unbiased = self.mb[l] / (1 - self.beta_1**self.adam_counter)
            v_unbiased = self.vb[l] / (1 - self.beta_2**self.adam_counter)
            self.b[l] -= self.lr * m_unbiased / (np.sqrt(v_unbiased) + self.epsilon)

    def forward(self, X):
        """Forward pass. Returns output layer and intermediate values in cache
        which will be used during backprop."""

        A = dict()
        Z = dict()
        D = dict()  # dropout coefficients

        A[0] = X
        for l in range(1, self.n_layers):
            Z[l] = self.W[l].dot(A[l - 1]) + self.b[l]
            A[l] = self.activations[l](Z[l])
            if l != self.n_layers - 1:
                # No dropout on input layer and output layer. In Hinton's
                # paper, they allow dropout for input layer but indicate it
                # should be very low.
                D[l] = np.random.binomial(n=1, p=self.dropout, size=A[l].shape)
                A[l] = A[l] * D[l] / self.dropout

        cache = A, Z, D

        return A[self.n_layers - 1], cache

    def backward(self, X, y, cache):
        """Backward pass. Returns gradients."""

        A, Z, D = cache

        n = X.shape[1]  # number of training examples

        dZ = dict()  # Note: no need to keep dA[l]
        dW = dict()
        db = dict()

        # Backprop last layer
        l = self.n_layers - 1
        dZ[l] = A[l] - y  # works with both log loss and cross-entropy
        dW[l] = 1 / n * (dZ[l].dot(A[l - 1].T) + self.lbd * self.W[l])
        db[l] = 1 / n * np.sum(dZ[l], axis=1, keepdims=True)

        # Backprop remaining layers
        for l in reversed(range(1, self.n_layers - 1)):
            dAl = self.W[l + 1].T.dot(dZ[l + 1])
            dAl = dAl * D[l] / self.dropout
            dAdZ = self.activations_deriv[l](Z[l])
            dZ[l] = dAdZ * dAl
            dW[l] = 1 / n * (dZ[l].dot(A[l - 1].T) + self.lbd * self.W[l])
            db[l] = 1 / n * np.sum(dZ[l], axis=1, keepdims=True)

        return dW, db

    def predict(self, X):
        """Predict outputs of entries in X [n_entries, n_features]"""
        y_hat, _ = self.forward(X.T)
        if self.n_neurons[-1] == 1:  # 2-class problem
            return (y_hat > .5).squeeze().astype('int')
        else:  # C > 2 class problem
            return np.argmax(y_hat, axis=0)

    def logistic_loss(self, y_hat, y):
        n = y_hat.shape[1]
        loss = - 1 / n  * np.sum(y * np.log(y_hat) +
                                      (1 - y) * np.log(1 - y_hat))
        loss += self.lbd / (2 * n) * np.sum([np.sum(W**2) for W in
                                             self.W.values()])
        return loss

    def cross_entropy_loss(self, y_hat, y):
        n = y_hat.shape[1]

        loss = - 1 / n * np.sum(y * np.log(y_hat))
        loss += self.lbd / (2 * n) * np.sum([np.sum(W**2) for W in
                                             self.W.values()])
        return loss


    def get_batches(self, X, y):
        """Return a list of batches (X_b, yb) to train on."""

        if self.batch_size == 0:
            return [(X, y)]  # don't do batch GD

        # Shuffle training set
        index = np.random.permutation(X.shape[1])
        X, y = X[:, index], y[:, index]
        # (it's actually number of batches - 1)
        n_batches = X.shape[1] // self.batch_size
        batches = []
        for k in range(n_batches):
            batches.append(
                (X[:, k * self.batch_size : (k + 1) * self.batch_size],
                 y[:, k * self.batch_size : (k + 1) * self.batch_size])
            )
        # Add remaining instances
        batches.append((X[:, n_batches * self.batch_size :],
                        y[:, n_batches * self.batch_size :]))
        return batches


    def check_gradients(self, X, y, dW, db):
        '''Do gradient checking for every single parameter. Raises an exception
        if computed gradients and estimated gradients are not close enough.'''

        epsilon = 1E-7

        for l, W in self.W.items():
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    W[i, j] += epsilon
                    lossplus = self.compute_loss(self.forward(X)[0], y)
                    W[i, j] -= 2 * epsilon
                    lossminus = self.compute_loss(self.forward(X)[0], y)

                    grad_approx = (lossplus - lossminus) / (2 * epsilon)
                    grad_computed = dW[l][i, j]

                    assert np.isclose(grad_approx, grad_computed)

                    W[i, j] += epsilon  # reset param to initial value

        for l, b in self.b.items():
            for i in range(b.shape[0]):
                b[i] += epsilon
                lossplus = self.compute_loss(self.forward(X)[0], y)
                b[i] -= 2 * epsilon
                lossminus = self.compute_loss(self.forward(X)[0], y)

                grad_approx = (lossplus - lossminus) / (2 * epsilon)
                grad_computed = db[l][i]

                assert np.isclose(grad_approx, grad_computed)

                b[i] += epsilon  # reset param to initial value
