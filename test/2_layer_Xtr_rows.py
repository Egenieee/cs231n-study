from keras.datasets import cifar10
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

    # ReLU 이용
    hidden_layer = np.maximum(0, np.dot(Xtr_rows, W) + b)
    scores = np.dot(hidden_layer, W2) + b2
    print("scores done")

    # softmax loss function
    exp_scores = np.exp(scores)
    print("exp_scores done")
    softmax = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    print("softmax done")

    # get loss
    correct_logprobs = -np.log(softmax[range(num_example), Ytr_rows])
    print("correct_logprobs done")
    data_loss = np.sum(correct_logprobs) / num_example
    reg_loss = 0.5 * reg * np.sum(W * W) + 0.5 * reg * np.sum(W2 * W2)
    loss = data_loss + reg_loss
    print("loss done")

    if i % 10 == 0:
        print("iteration %d: loss %f" % (i, loss))

    # scores에 대한 점수 구하기
    dscores = softmax
    dscores[range(num_example), Ytr_rows] -= 1
    dscores /= num_example

    # W2와 b2에 backprop
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)

    dhidden = np.dot(dscores, W2.T)

    dhidden[hidden_layer <= 0] = 0

    dW = np.dot(Xtr_rows.T, dhidden)
    db = np.sum(dhidden, axis=0, keepdims=True)

    dW2 += reg * W2
    dW += reg * W

    W += -step_size * dW
    b += -step_size * db
    W2 += -step_size * dW2
    b2 += -step_size * db2

hidden_layer = np.maximum(0, np.dot(Xtr_rows, W) + b)
scores = np.dot(hidden_layer, W2) + b2
tr_pred_class = np.argmax(scores, axis=0)
print("training accuracy: %.2f" % (np.mean(tr_pred_class == Ytr_rows)))

hidden_layer = np.maximum(0, np.dot(Xte_rows, W) + b)
scores = np.dot(hidden_layer, W2) + b2
te_pred_class = np.argmax(scores, axis=0)
print("test accuracy: %.2f" % (np.mean(te_pred_class == Yte_rows)))