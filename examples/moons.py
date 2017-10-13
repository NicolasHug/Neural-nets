import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

from plot_decision import plot_decision_boundaries
from neuralnets import NeuralNet


X, y = make_moons(n_samples=100, noise=.1)

seed = None
nn = NeuralNet(n_neurons=[X.shape[1], 5, 1], activations='tanh',
               seed=seed, learning_rate=1.2, n_epochs=2000)

nn.fit(X, y)

plt.figure()
plt.plot(nn.losses)
plt.title('Loss vs number of epochs')

plt.figure()
plot_decision_boundaries(nn, X, y)
plt.title('Decision boundaries')

plt.show()
