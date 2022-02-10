from keras.datasets import cifar10
from gradient import following_gradient as fg
import numpy as np

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# flatten out all images to be one-dimensional
Xtr_cols = X_train.reshape(32 * 32 * 3, X_train.shape[0]) # Xtr_rows becomes 3072 x 50000
Xte_cols = X_test.reshape(32 * 32 * 3, X_test.shape[0]) # Xte_cols becomes 3072 x 10000

Ytr_cols = y_train.reshape(1, y_train.shape[0]) # Ytr_cols becomes 1 x 50000
Yte_cols = y_test.reshape(1, y_test.shape[0]) # Yte_cols becomes 1 x 10000


W = np.random.randn(10, 3072) * 0.0001
b = np.zeros((10,1))
reg = 1e-3

step_size = 10 ** -6.8

for i in range(100):
    # loss function을 통해 loss, dW, db를 구함
    loss, dW, db = fg.get_gradient_softmax(Xtr_cols, Ytr_cols, W, b, reg)

    if i % 10 == 0:
        print("현재 {}번째 시도 loss는 {}".format(i, loss))

    W += -step_size * dW
    b += -step_size * db

scores = np.dot(W, Xtr_cols) + b
tr_pred_class = np.argmax(scores, axis=0)
print("training accuracy: %.2f" % (np.mean(tr_pred_class == Ytr_cols)))

scores = np.dot(W, Xte_cols) + b
te_pred_class = np.argmax(scores, axis=0)
print("test accuracy: %.2f" % (np.mean(te_pred_class == Yte_cols)))
