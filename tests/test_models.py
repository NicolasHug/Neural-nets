import numpy as np
from sklearn.datasets import make_blobs

from neuralnets import NeuralNet
from neuralnets import LogisticReg
from neuralnets import TwoLayersNN


def test_equivalent_models():
    '''Ensures that NeuralNet gives the same results as TwoLayersNN and
    LogisticReg when used with the same settings'''

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

