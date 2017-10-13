import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

from plot_decision import plot_decision_boundaries
from neuralnets import LogisticReg
from neuralnets import NeuralNet


X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=0)

clf = LogisticReg()
nn = NeuralNet(n_neurons=[X.shape[1], 5, 1])

clf.fit(X, y)
nn.fit(X, y)

plt.figure()
plt.plot(clf.losses, label='Logistic reg')
plt.plot(nn.losses, label='NN')
plt.legend()
plt.title('Loss vs number of epoch')

plt.figure()
plot_decision_boundaries(clf, X, y)
plot_decision_boundaries(nn, X, y)
plt.title('Decision boundaries')
plt.show()
