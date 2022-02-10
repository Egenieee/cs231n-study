from keras.datasets import cifar10
import loss_function as lf
import numpy as np

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# flatten out all images to be one-dimensional
Xtr_rows = X_train.reshape(X_train.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xtr_cols = X_train.reshape(32 * 32 * 3, X_train.shape[0]) # Xtr_rows becomes 3072 x 50000
Ytr_cols = y_train.reshape(1, y_train.shape[0]) # Ytr_cols becomes 1 x 50000

# print("X_train.shape : {}".format(X_train.shape))
# print("X_test.shape : {}".format(X_test.shape))
# print("y_train.shape : {}".format(y_train.shape))
# print("y_test.shape : {}".format(y_test.shape))
# print("Xtr_rows.shape : {}".format(Xtr_rows.shape))
# print("Xtr_cols.shape : {}".format(Xtr_cols.shape))
# print("Ytr_cols.shape : {}".format(Ytr_cols.shape))

print("---------------------loss function test------------------------")

W = np.random.randn(10, 3072) * 0.0001
b = np.zeros((10, 1))
reg = 1e-3 # reguluarization strength

print("입력받은 이미지로 SVM을 이용하여 Loss를 측정합니다...")
print()

svm_loss = lf.L_SVM(Xtr_cols, Ytr_cols, W, b, reg)

print("svm_loss : {}".format(svm_loss))

print()

print("입력받은 이미지로 Softmax을 이용하여 Loss를 측정합니다...")
print()

softmax_loss = lf.L_Softmax(Xtr_cols, Ytr_cols, W, b, reg)

print("softmax_loss : {}".format(softmax_loss))


