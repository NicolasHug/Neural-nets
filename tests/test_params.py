import numpy as np
from sklearn.datasets import make_blobs

from neuralnets import NeuralNet


def test_reg():
    '''Ensures that regularization is taken into account.'''

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


def test_init_strat():
    '''Ensure that weights are not the same with He initialization'''

    seed = 0
    nn1 = NeuralNet(n_neurons=[2, 2, 3],
                    init_strat='He',
                    seed=seed)
    nn2 = NeuralNet(n_neurons=[2, 2, 3],
                    init_strat='None',
                    seed=seed)
    assert (nn1.W[1] != nn2.W[1]).any()
    assert (nn1.W[2] != nn2.W[2]).any()

def test_adam_sgd():
    '''Ensure that using adam gives different results than sgd'''

    learning_rate = .005
    n_epochs = 10
    seed = 0

    X, y = make_blobs(n_samples=100, centers=3, n_features=2,
                      random_state=seed)

    nn1 = NeuralNet(n_neurons=[2, 2, 3],
                    learning_rate=learning_rate,
                    n_epochs=n_epochs,
                    solver='sgd',
                    seed=seed)
    nn2 = NeuralNet(n_neurons=[2, 2, 3],
                    learning_rate=learning_rate,
                    n_epochs=n_epochs,
                    solver='adam',
                    seed=seed)
    nn1.fit(X, y)
    nn2.fit(X, y)
    assert nn1.losses != nn2.losses
