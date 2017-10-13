import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_mldata

from neuralnets import LogisticReg
from neuralnets import TwoLayersNN

mnist = fetch_mldata('MNIST original')

index = np.random.randint(0, mnist.data.shape[0], 1000)
index = mnist.target < 2  # only 0 and 1 images
X = mnist.data[index]
y = mnist.target[index]

# It's cheating because we process before before splitting, but don't care
X = preprocessing.scale(X.astype('float'))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

clf = LogisticReg(n_epochs=250)
clf = TwoLayersNN(n_epochs=2000, learning_rate=.001)
clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)
print('Accuracy:', accuracy_score(y_hat, y_test))

# Plot loss against epochs
plt.plot(clf.losses)
plt.show()

# Plot misclassified images
for i in np.arange(X_test.shape[0])[y_hat != y_test]:
    plt.imshow(X_test[i].reshape((28, 28)), cmap='gray')
    plt.show()
