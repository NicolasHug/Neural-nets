import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

from plot_decision import plot_decision_boundaries
from neuralnets import NeuralNet


n_blobs = 4
X, y = make_blobs(n_samples=100, centers=n_blobs)

seed = None
nn = NeuralNet(n_neurons=[X.shape[1], 5, n_blobs], activations='tanh',
               seed=seed, learning_rate=1.2, n_epochs=2000)

nn.fit(X, y)

plt.figure()
plt.plot(nn.losses)
plt.title('Loss vs number of epochs')

plt.figure()
plot_decision_boundaries(nn, X, y)
plt.title('Decision boundaries')

plt.show()
