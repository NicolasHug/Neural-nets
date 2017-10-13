import numpy as np
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons

from neuralnets import NeuralNet
from neuralnets import LogisticReg
from neuralnets import TwoLayersNN


def test_equivalent_models():
    """Ensures that NeuralNet gives the same results as TwoLayersNN and
    LogisticReg when used with the same settings"""

    learning_rate = .005
    n_epochs = 500

    seed = 0
    X, y = make_blobs(n_samples=1000, centers=2, n_features=2,
                      random_state=seed)

    # NN vs TwoLayers
    two = TwoLayersNN(learning_rate=learning_rate, activation='tanh',
                         n_epochs=n_epochs, seed=seed)
    nn = NeuralNet(n_neurons=[2, 5, 1], activations='tanh',
                   learning_rate=learning_rate, n_epochs=n_epochs,
                   batch_size=0, seed=seed)
    two.fit(X, y)
    nn.fit(X, y)

    # indexes differ because loss is computed at the END of each epoch for the
    # NeuralNet class
    assert np.allclose(two.losses[1:], nn.losses[:-1])
    assert np.allclose(two.W1, nn.W[1])
    assert np.allclose(two.W2, nn.W[2])
    assert np.allclose(two.b1, nn.b[1])
    assert np.allclose(two.b2, nn.b[2])

    # NN vs LogisticReg
    log_reg = LogisticReg(learning_rate=learning_rate, n_epochs=n_epochs,
                      seed=seed)
    nn = NeuralNet(n_neurons=[2, 1], learning_rate=learning_rate,
                   n_epochs=n_epochs, batch_size=0, seed=seed)

    log_reg.fit(X, y)
    nn.fit(X, y)
    assert np.allclose(log_reg.losses[1:], nn.losses[:-1])
    assert np.allclose(log_reg.w.squeeze(), nn.W[1])
    assert np.allclose(log_reg.b, nn.b[1])


def test_reg():
    """Ensures that regularization is taken into account."""

    n_epochs = 1000

    seed = 0
    X, y = make_blobs(n_samples=100, centers=3, n_features=2,
                      random_state=seed)

    nn_reg = NeuralNet(n_neurons=[2, 2, 3],
                   activations='tanh',
                   n_epochs=n_epochs,
                   lambda_reg=1,
                   seed=seed)
    nn_no_reg = NeuralNet(n_neurons=[2, 2, 3],
                   activations='tanh',
                   n_epochs=n_epochs,
                   lambda_reg=0,
                   seed=seed)
    nn_reg.fit(X, y)
    nn_no_reg.fit(X, y)

    assert nn_reg.losses != nn_no_reg.losses
    assert nn_reg.losses[-1] > nn_no_reg.losses[-1]  # reg hurts train set loss


def test_check_gradients():
    """Gradient-check for NN on various datasets, architecture and loss
    functions, with batch learning and regularization.
    """

    learning_rate = .005
    n_epochs = 1000
    batch_size = 64
    lambda_reg = .1

    seed = 0

    X, y = make_blobs(n_samples=100, centers=3, n_features=2,
                      random_state=seed)
    nn = NeuralNet(n_neurons=[2, 2, 3],
                   activations='tanh',
                   learning_rate=learning_rate,
                   n_epochs=n_epochs,
                   batch_size=64,
                   lambda_reg=lambda_reg,
                   check_gradients=True,
                   seed=seed)
    nn.fit(X, y)  # would raise exception if grad check fails

    X, y = make_moons(n_samples=100, noise=.1, random_state=seed)
    nn = NeuralNet(n_neurons=[2, 2, 2, 1],
                   activations='tanh',
                   learning_rate=learning_rate,
                   n_epochs=n_epochs,
                   batch_size=64,
                   lambda_reg=lambda_reg,
                   check_gradients=True,
                   seed=seed)
    nn.fit(X, y)  # would raise exception if grad check fails
