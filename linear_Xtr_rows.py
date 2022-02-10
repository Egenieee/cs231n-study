from keras.datasets import cifar10
import following_gradient as fg
import numpy as np

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# flatten out all images to be one-dimensional
Xtr_rows = X_train.reshape(X_train.shape[0], 32 * 32 * 3) # 50000 x 3072
Xte_rows = X_test.reshape( X_test.shape[0], 32 * 32 * 3) # 10000 x 3072

Ytr_rows = y_train.reshape(y_train.shape[0], 1) # 50000 x 1
Yte_rows = y_test.reshape(y_test.shape[0], 1) # 10000 x 1

print("Xtr_rows.shape : {}".format(Xtr_rows.shape))
print("Ytr_rows.shape : {}".format(Ytr_rows.shape))

# hidden layer =  100
W = np.random.randn(3072, 100) * 0.0001
b = np.zeros((1, 100))
W2 = np.random.randn(100, 10)
b2 = np.zeros((1, 10))

step_size = 1e-3
reg = 1e-3

num_example = Xtr_rows.shape[0]
print("number of examples is {}".format(num_example))

for i in range(1000):
    # W와 X를 내적하여 scores 행렬 구함. scores의 dim = 10 x 50000
    scores = Xtr_rows.dot(W) + b

    exp_scores = np.exp(scores)

    sum_exp_scores = np.exp(scores).sum(axis=1, keepdims=True)
    # print("sum_exp_scores.shape : {}".format(sum_exp_scores.shape))
    # exp_scores : 10 x 50000
    # sum_exp_score : 1 x 50000

    softmax = exp_scores / sum_exp_scores
    # print("softmax.shape : {}".format(softmax.shape))

    # 정답 클래스의 값에 -log를 취한다.
    loss_i = -np.log(softmax[range(num_example), Ytr_rows])

    # 평균 data loss
    data_loss = np.sum(loss_i) / num_example

    # regularization loss
    reg_loss = 0.5 * reg * np.sum(W * W)

    # 최종 loss
    loss = data_loss + reg_loss

    if i % 10 == 0:
        print("iteration %d: loss %f" % (i, loss))

    # -----------------get gradient-----------------#

    # softmax에는 모든 클래스에 대한 확률이 저장되어 있다. loss를 이제 미분한다.
    dscores = softmax
    # 정답 클래스의 확률에서만 1을 빼준다.
    dscores[range(num_example), Ytr_rows] -= 1
    dscores /= num_example

    # backpropate the gradient to the parameters (W, b)
    dW = dscores.dot(Xtr_rows.T)
    db = np.sum(dscores, axis=0, keepdims=True)

    dW += reg * W  # regularization gradient

    W += -step_size * dW
    b += -step_size * db

scores = np.dot(Xtr_rows, W) + b
tr_pred_class = np.argmax(scores, axis=0)
print("training accuracy: %.2f" % (np.mean(tr_pred_class == Ytr_rows)))

scores = np.dot(Xte_rows, W) + b
te_pred_class = np.argmax(scores, axis=0)
print("test accuracy: %.2f" % (np.mean(te_pred_class == Yte_rows)))