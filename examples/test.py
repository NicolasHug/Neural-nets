import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score

from neuralnets import TwoLayersNN
from neuralnets import LogisticReg
from neuralnets import NeuralNet


def plot_decision_boundaries(clf, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')


#X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=0)
#X, y = make_moons(n_samples=100, noise=.1)
X, y = make_circles(n_samples=1000, noise=.05)

seed = None
#two = TwoLayersNN(n_h=5, seed=seed, n_epochs=10000)
nn = NeuralNet(n_neurons=[X.shape[1], 5, 5, 1], activation='tanh',
               seed=seed, learning_rate=.9, n_epochs=20000)

print(nn.activations)
print(nn.activations_deriv)
#two.fit(X, y)
nn.fit(X, y)

plt.plot(nn.losses)
#plt.plot(two.losses)

plt.figure()
#plot_decision_boundaries(two, X, y)
plot_decision_boundaries(nn, X, y)

plt.show()


print(accuracy_score(y, nn.predict(X)))
