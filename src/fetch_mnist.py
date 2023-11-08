import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def fetch_mnist():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, Y = mnist["data"], mnist["target"]
    # normalize to [0, 1]
    X, Y = X/255.0, Y.astype(np.int8)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=7, test_size=1/7)
    return X_train, X_test, Y_train, Y_test
