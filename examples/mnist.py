import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from neuralnets import NeuralNet

mnist = fetch_mldata('MNIST original')

# Subsample mnist dataset: only keep 1000 images
n_images = 1000
index = np.random.randint(0, mnist.data.shape[0], n_images)
X = mnist.data[index]
y = mnist.target[index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
X_train = preprocessing.scale(X_train.astype('float'))
X_test = preprocessing.scale(X_test.astype('float'))

clf = NeuralNet(n_neurons=[X_train.shape[1], 5, 10],
                activations='relu',
                n_epochs=20000,
                verbose=True)
clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)
print('Accuracy: {0:1.3f}'.format(accuracy_score(y_hat, y_test)))

# Plot loss against epochs
plt.plot(clf.losses)
plt.show()

# Plot misclassified images
for i in np.arange(X_test.shape[0])[y_hat != y_test]:
    plt.imshow(X_test[i].reshape((28, 28)), cmap='gray')
    plt.show()
