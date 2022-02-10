from keras.datasets import cifar10
from loss import loss_function as lf
import numpy as np

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# flatten out all images to be one-dimensional
Xtr_rows = X_train.reshape(X_train.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xtr_cols = X_train.reshape(32 * 32 * 3, X_train.shape[0]) # Xtr_rows becomes 3072 x 50000
Xte_cols = X_test.reshape(32 * 32 * 3, X_test.shape[0]) #Xte_cols becomes 3072 x 10000

Ytr_cols = y_train.reshape(1, y_train.shape[0]) # Ytr_cols becomes 1 x 50000
Yte_cols = y_test.reshape(1, y_test.shape[0]) # Yte_cols becomes 1 x 10000

print("---------------------Optimization #1 : Random Search------------------------")

# 가장 큰 float값 할당
best_loss = float("inf")
best_W = 0

for num in range(1000):
    # W에는 random 값 넣어줌

    W = np.random.randn(10, 3072) * 0.0001
    b = np.zeros((10, 1))
    reg = 1e-3

    # SVM loss function을 이용하여 loss를 구함
    svm_loss = lf.L_SVM(Xtr_cols, Ytr_cols, W, b, reg)

    # 현재 구한 loss값이 지금까지 나왔던 제일 작은 loss와 비교하고 작으면 제일 작은 loss값 업데이트
    if svm_loss < best_loss:
        best_loss = svm_loss
        # 작은 loss를 만든 W도 beat_W로 업데이트
        best_W = W
    print ('in attempt %d the loss was %f, best %f' % (num, svm_loss, best_loss))

# 위에서 랜덤 서치로 구한 W로 test set을 test한다
scores = best_W.dot(Xte_cols)
Yte_predict = np.argmax(scores, axis=0)

# 정확도 출력
print("accuracy : {}".format(np.mean(Yte_predict == Yte_cols)))