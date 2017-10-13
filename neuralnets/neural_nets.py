import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder

from . import utils

class NeuralNet(BaseEstimator):
    """A basic neural network class.

    Args:
        n_neurons (list):list of H + 2 integers indicating the number of
            neurons in each layer, including input and output layers. First
            value is the number of features of a training example. Last value
            is either 1 (logistic loss) or C > 1 (cross entropy), where C is
            the number of classes. In-between, the H values indicate the number
            of neurons of each of the H hidden layer.
        activations (str or list of str): The activation functions to use for
            each of the H hidden layers. Allowed values are 'sigmoid', 'tanh',
            'relu' or 'linear' (i.e. no activation). If a str is given, then
            all activations are the same for each hidden layer. If a list of
            string is given, it must be of size H. Default is 'relu'. Note: the
            activation function of the last layer is automatically inferred
            from the value of n_neurons[-1]: if the output layer size is 1 then
            a sigmoid is used, else it's a softmax.
        learning_rate(float): The learning rate for gradient descent.
        n_epochs(int): The number of iteration of the gradient descent
            procedure, i.e. number of times the whole training set is gone
            through.
        batch_size(int): The batch size. If 0, the full trainset is used.
        lambda_reg(float): The regularization constant. Default is 0, i.e. no
            regularization.
        init_strat(str): Initialization strategy for weights. Can be 'He' for
            'He' initialization, recommended for relu layers. Default is None,
            which reverts to a centered normal distribution * 0.1.
        seed(int): A random seed to use for the RNG.
        check_gradients(bool): Whether to check gradients at each iteration,
            for each parameter. It's done with np.isclose() with default
            tolerance values. Default is False.
        verbose(int): if not False or 0, will print the loss every 'verbose'
            epochs. Default is False.


    Note: The NeuralNet estimator is compliant with scikit-learn API so the
    inputs X fit and predict is [n_entries, n_features], but internally we use
    X.T because it seems more convenient.
    """

    def __init__(self, n_neurons, activations='relu', learning_rate=.005,
                 n_epochs=10000, batch_size=64, lambda_reg=0, init_strat=None,
                 seed=None, check_gradients=False, verbose=False):

        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.n_neurons = n_neurons
        self.n_layers = len(self.n_neurons)  # including input and output
        self.m = n_neurons[0]  # dimension of input layer
        self.losses = []
        self.do_grad_check = check_gradients
        self.verbose = verbose
        self.batch_size = batch_size
        self.lbd = lambda_reg
        self.init_strat = init_strat

        if n_neurons[-1] == 1:
            self.compute_loss = self.logistic_loss
        else:
            self.compute_loss = self.cross_entropy_loss

        self.init_weights(seed)
        self.init_activations(activations)

    def init_activations(self, activations):

        if isinstance(activations, str):  # transform str into list of same str
            activations = [activations] * (self.n_layers - 2)
        activations_dict = {  # map name to functions
            'sigmoid': (utils.sigmoid, utils.sigmoid_deriv),
            'relu': (utils.relu, utils.relu_deriv),
            'tanh': (utils.tanh, utils.tanh_deriv),
            'linear': (utils.linear, utils.linear_deriv),
        }
        self.activations = dict()
        self.activations_deriv = dict()
        for i, activation in enumerate(activations):
            try:
                act_fun, deriv_fun = activations_dict[activation]
                self.activations[i + 1] = act_fun
                self.activations_deriv[i + 1] = deriv_fun
            except KeyError:
                exit('Unsupported activation' + activation)

        # Last layer: either sigmoid or softmax
        if self.n_neurons[-1] == 1:
            self.activations[self.n_layers - 1] = utils.sigmoid
            self.activations_deriv[self.n_layers - 1] = utils.sigmoid_deriv
        else:
            self.activations[self.n_layers - 1] = utils.softmax
            self.activations_deriv[self.n_layers - 1] = utils.softmax_deriv

    def init_weights(self, seed):
        if seed is not None:
            np.random.seed(seed)
        self.W = dict()
        self.b = dict()
        for l in range(1, self.n_layers):
            if self.init_strat == 'He':
                self.W[l] = (np.random.randn(self.n_neurons[l],
                                             self.n_neurons[l - 1]) *
                             np.sqrt(2 / self.n_neurons[l - 1]))
            else:
                self.W[l] = np.random.randn(self.n_neurons[l],
                                             self.n_neurons[l - 1]) * .01

            self.b[l] = np.zeros((self.n_neurons[l], 1))

    def fit(self, X, y):
        """
        Fit model with input X[n_entries, n_features] and output y.
        """

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

                for l in range(1, self.n_layers):
                    self.W[l] -= self.lr * dW[l]
                    self.b[l] -= self.lr * db[l]

            # Compute loss at the end of each epoch
            y_hat, _ = self.forward(X)
            loss = self.compute_loss(y_hat, y)
            self.losses.append(loss)
            if self.verbose and current_epoch % self.verbose == 0:
                print('Epoch {0:5d}, loss= {1:1.3f}'.format(current_epoch,
                                                            loss))


    def forward(self, X):
        """Forward pass. Returns output layer and intermediate values in cache
        which will be used during backprop."""

        A = dict()
        Z = dict()

        A[0] = X
        for l in range(1, self.n_layers):
            Z[l] = self.W[l].dot(A[l - 1]) + self.b[l]
            A[l] = self.activations[l](Z[l])

        cache = A, Z  # weights are attributes so no need to cache them

        return A[self.n_layers - 1], cache

    def backward(self, X, y, cache):
        """Backward pass. Returns gradients."""

        A, Z = cache
        n = X.shape[1]  # number of training examples

        dZ = dict()  # Note: no need to keep dA[l]
        dW = dict()
        db = dict()

        # Backprop last layer
        l = self.n_layers - 1
        dZ[l] = A[l] - y
        dW[l] = 1 / n * (dZ[l].dot(A[l - 1].T) + self.lbd * self.W[l])
        db[l] = 1 / n * np.sum(dZ[l], axis=1, keepdims=True)

        # Backprop remaining layers
        for l in reversed(range(1, self.n_layers - 1)):
            dAl = self.W[l + 1].T.dot(dZ[l + 1])
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
