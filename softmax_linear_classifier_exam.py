import numpy as np
import matplotlib.pyplot as plt

# generating some data

N = 100 # 클래스 하나당 점의 개수
D = 2 # 차원
K = 3 # 클래스 개수
X = np.zeros((N*K, D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels

print("X.shape : {}".format(X.shape))
print("y.shape : {}".format(y.shape))

# 점들을 random하게 흩뿌려 주자!

for j in range(K):
    ix = range(N*j , N*(j + 1))
    r = np.linspace(0.0,1,N) # radius
    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N) * 0.2 # theta
    X[ix] = np. c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

# lets visualize the data
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()

# train a linear classifier

# initialize parameters randomly
W = 0.01 * np.random.randn(D, K) # 2 x 3
b = np.zeros((1, K)) # 1 x 3

# some hyperparameters
step_size = 1e-0
reg = 1e-3 # reguluarization strength

# gradient descent loop
num_examples = X.shape[0] # 300개
print("num_examples is {}".format(num_examples))
for i in range(10000):

    # evaluate class scores, [N x K] 300 x 3
    scores = np.dot(X, W) + b

    # compute the class probabilites
    exp_score = np.exp(scores)
    probs = exp_score / np.sum(exp_score, axis=1, keepdims=True) # [N x K] 300 x 3

    # compute the loss : average cross-entropy loss ans regularization
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs) / num_examples
    reg_loss = 0.5 * reg * np.sum(W * W)
    loss = data_loss + reg_loss

    if i % 30 == 0:
        print("iteration %d: loss %f" % (i, loss))

    # compute the gradient on scores
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples

    # backpropate the gradient to the parameters (W, b)
    dW = np.dot(X.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)

    dW += reg*W # regularization gradient

    # perform a parameter gradient
    W += -step_size * dW
    b += -step_size * db

# evaluate training set accuracy
scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print("training accuracy: %.2f" % (np.mean(predicted_class == y)))

h = 0.02
x_min = X[:, 0].min() - 1
x_max = X[:, 0].max() + 1
y_min = X[:, 1].min() - 1
y_max = X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.6)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, alpha=0.9, edgecolors='black')
plt.show()

# # Setting up input values
# Sx = np.arange(-2.0, 2.0, 0.1)
# Sy = np.arange(-2.0, 2.0, 0.1)
# SX, SY = np.meshgrid(Sx, Sy)


# SY[0:3, :] = 0
# SY[10:20, :] = 1
# SY[20:, :] = 2


# # plot heatmap colorspace in the background
# fig, ax = plt.subplots(nrows=1)
# im = ax.imshow(SY, cmap=plt.cm.Spectral, extent=(-2, 2, -2, 2), interpolation='bilinear')
#
# # lets visualize the data
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolors='black')
# plt.show()