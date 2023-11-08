import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def fetch_mnist():
    mnist = fetch_openml("mnist_784", version=1)
    X, Y = mnist["data"], mnist["target"]
    # norm to [0, 1]
    X, Y = X/255, Y.astype(np.int8)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=7, test_size=1/7)
    return X_train, X_test, Y_train, Y_test