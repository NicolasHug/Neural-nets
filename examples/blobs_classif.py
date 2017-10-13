import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

from neuralnets import LogisticReg


def plot_decision_boundaries(clf, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')


X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=0)

clf = LogisticReg()
clf.fit(X, y)
plt.figure()
plt.plot(clf.losses)

plt.figure()
plot_decision_boundaries(clf, X, y)
plt.show()
