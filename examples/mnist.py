import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from neuralnets import NeuralNet

mnist = fetch_mldata('MNIST original')
X, y = mnist.data, mnist.target
# Original mnist split
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

X_train = preprocessing.scale(X_train.astype('float'))
X_test = preprocessing.scale(X_test.astype('float'))

clf = NeuralNet(n_neurons=[X_train.shape[1], 100, 100, 10],
                activations='relu',
                learning_rate=.001,
                n_epochs=20,
                batch_size=128,
                lambda_reg=.4,
                init_strat='He',
                solver='adam',
                verbose=10)

clf.fit(X_train, y_train)
y_hat = clf.predict(X_train)
print('Accuracy on trainset: {0:1.3f}'.format(accuracy_score(y_hat, y_train)))
y_hat = clf.predict(X_test)
print('Accuracy on testset:  {0:1.3f}'.format(accuracy_score(y_hat, y_test)))

plot_misclassified = input('Do you want to see the misclassified images? ([y] / n)')
if plot_misclassified.lower() in ('', 'y', 'yes'):
    for i in np.arange(X_test.shape[0])[y_hat != y_test]:
        print('I thought it was a ' + str(y_hat[i]))
        plt.imshow(X_test[i].reshape((28, 28)), cmap='gray')
        plt.show(block=False)
        input('Press Enter for next one')
