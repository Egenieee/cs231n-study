from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# flatten out all images to be one-dimensional
Xtr_cols = X_train.reshape(32 * 32 * 3, X_train.shape[0]) # Xtr_rows becomes 3072 x 50000
Xte_cols = X_test.reshape(32 * 32 * 3, X_test.shape[0]) # Xte_cols becomes 3072 x 10000

Ytr_cols = y_train.reshape(1, y_train.shape[0]) # Ytr_cols becomes 1 x 50000
Yte_cols = y_test.reshape(1, y_test.shape[0]) # Yte_cols becomes 1 x 10000

# hidden layer =  100
W = np.random.randn(100, 3072) * 0.0001 # 100 x 3072
b = np.zeros((100,1)) # 100 x 1
W2 = np.random.randn(10, 100) # 10 x 100
b2 = np.zeros((10, 1)) # 10 x 1

step_size = 10 ** -6.8
reg = 1e-3

best_loss = float("inf")
loss_plot = []
num_iter = np.arange(1000)

num_example = Xtr_cols.shape[1]
print("number of examples is {}".format(num_example))

print("softmax function을 이용한 2 layer neural network로 학습한 결과")
print()

for i in range(1000):

    # ReLU 이용
    hidden_layer = np.maximum(0, W.dot(Xtr_cols) + b) # 100 x 50000
    scores = W2.dot(hidden_layer) + b2 # 10 x 50000

    # softmax loss function
    exp_scores = np.exp(scores) # 10 x 50000
    softmax = exp_scores / np.sum(exp_scores, axis=0, keepdims=True) # 10 x 50000

    # get loss
    correct_logprobs = -np.log(softmax[Ytr_cols, range(num_example)])
    data_loss = np.sum(correct_logprobs) / num_example
    reg_loss = 0.5 * reg * np.sum(W * W) + 0.5 * reg * np.sum(W2 * W2)
    loss = data_loss + reg_loss
    loss_plot.append(loss)

    if loss < best_loss:
        best_loss = loss

    if i % 10 == 0:
        print("{}번째 시도, loss는 {}".format(i, loss))
        print("현재까지 best loss는 {}".format(best_loss))
        print()

    # scores에 대한 gradient 구하기
    dscores = softmax
    dscores[Ytr_cols, range(num_example)] -= 1
    dscores /= num_example

    # W2와 b2에 backprop
    # upstream gradient = dscores, local gradient = hidden_layer
    dW2 = dscores.dot(hidden_layer.T) # 10 x 100
    db2 = np.sum(dscores, axis=1, keepdims=True) # 10 x 1

    # upstream gradient = dscores, local gradient = W2
    dhidden = W2.T.dot(dscores)

    # ReLU 미분
    dhidden[hidden_layer <= 0] = 0

    # upstream gradient = dhidden, local gradient = Xtr_cols
    dW = dhidden.dot(Xtr_cols.T)
    db = np.sum(dhidden, axis=1, keepdims=True)

    dW2 += reg * W2
    dW += reg * W

    W += -step_size * dW
    b += -step_size * db
    W2 += -step_size * dW2
    b2 += -step_size * db2

plt.plot(num_iter, loss_plot)
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()

hidden_layer = np.maximum(0, W.dot(Xtr_cols) + b)
scores = W2.dot(hidden_layer) + b2
tr_pred_class = np.argmax(scores, axis=0)
print("training accuracy: %.2f" % (np.mean(tr_pred_class == Ytr_cols)))

hidden_layer = np.maximum(0, W.dot(Xte_cols) + b)
scores = W2.dot(hidden_layer) + b2
te_pred_class = np.argmax(scores, axis=0)
print("test accuracy: %.2f" % (np.mean(te_pred_class == Yte_cols)))