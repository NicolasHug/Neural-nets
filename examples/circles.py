"""Embedded circles classification problem."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

from plot_decision import plot_decision_boundaries
from neuralnets import NeuralNet


X, y = make_circles(n_samples=1000, noise=.05)

seed = None
nn = NeuralNet(n_neurons=[X.shape[1], 5, 1], activations='relu',
               seed=seed, learning_rate=.9, n_epochs=2000)

nn.fit(X, y)

plt.figure()
plt.plot(nn.losses)
plt.title('Loss vs number of epochs')

plt.figure()
plot_decision_boundaries(nn, X, y)
plt.title('Decision boundaries')

plt.show()
