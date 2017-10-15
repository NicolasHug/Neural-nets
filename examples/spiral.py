import numpy as np
import matplotlib.pyplot as plt

from plot_decision import plot_decision_boundaries
from neuralnets import NeuralNet


def generate_spiral(n, C):
    '''Generate n points forming a spiral with C classes. Taken from cs231n
    course notes.'''
    X = np.zeros((n * C, 2))
    y = np.zeros(n * C, dtype='uint8')
    for j in range(C):
        ix = range(n * j, n * (j + 1))
        r = np.linspace(0.0, 1, n) # radius
        t = np.linspace(j*4,(j + 1) * 4, n) + np.random.randn(n)*0.2 # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    return X, y


C = 3
X, y = generate_spiral(100, C)

clf = NeuralNet(n_neurons=[X.shape[1], 100, C],
                activations='relu',
                learning_rate=.1,
                n_epochs=20,
                batch_size=0,
                lambda_reg=.001,
                init_strat='He',
                solver='adam',
                verbose=2)

clf.fit(X, y)

plt.figure()
plot_decision_boundaries(clf, X, y)
plt.show()
