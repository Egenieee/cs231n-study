from keras.datasets import cifar10
import numpy as np

def eval_numerical_gradient(f, x):
    """
      a naive implementation of numerical gradient of f at x
      - f should be a function that takes a single argument
      - x is the point (numpy array) to evaluate the gradient at
      """
    print(x.shape)
    fx = f(x) # evaluate function value at original point
    gradient = np.zeros(x.shape, dtype=np.object_)
    h = 0.00001

    # iterate over all indexed in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # evaluate function at x+h
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h  # increment by h
        fxh = f(x)  # evalute f(x + h)
        x[ix] = old_value  # restore to previous value (very important!)

        # compute the partial derivative
        gradient[ix] = (fxh - fx) / h  # the slope
        it.iternext()  # step to next dimension

    return gradient

def get_gradient_svm(X, y, W, b, reg):
  # X는 전체 5만개의 트레이닝 셋, y는 50000개의 레이블 배열, W는 가중치

  # W와 X를 내적하여 scores 행렬 구함. scores의 dim = 10 x 50000
  scores = W.dot(X) + b
  #print("scores : {}".format(scores))
  # print("scores.shape : {}".format(scores.shape))

  # scores에서 정답인 레이블의 점수만 가져옴 dim = 1 x 50000
  # scores[y, range(X.shape[0])]의 뜻은 scores 배열에서 정답 클래스인 y번째 행과 0부터 시작해서 49999개의 열에서만 scores를 가져오겠다는 뜻.
  correct_class_scores = scores[y, range(X.shape[1])]
  #print("correct_class_scores : {}".format(correct_class_scores))
  # print("correct_class_scores.shape : {}".format(correct_class_scores.shape))

  # scores에 더해줘야 하는 delta값. scores의 dim과 같은 차원이다 dim = 10 x 50000
  delta = np.ones(scores.shape)

  # scores배열에서 정답 클래스의 점수를 뺀 뒤 delta값을 더해 margins라는 loss값을 담을 배열을 새로 만든다. dim = 10 x 50000
  margins = np.maximum(0, scores - correct_class_scores + delta)

  # 정답 클래스의 loss는 제외시키기 때문에 0으로 초기화
  margins[y, range(X.shape[1])] = 0

  # print("margin.shape : {}".format(margins.shape))

  ## 각 사진 마다의 loss를 구함
  loss_i = np.sum(margins, axis=0)

  # 평균 data loss
  data_loss = np.sum(loss_i) / X.shape[1]

  # regularization loss 구함
  reg_loss = 0.5 * reg * np.sum(W * W)

  # 최종 loss 구함
  loss = data_loss + reg_loss

  # print("margin.shape : {}".format(loss_i.shape))

  # -----------------get gradient-----------------#

  # loss가 담긴 행렬을 이제 미분한다.
  dscores = margins

  # 0보다 큰 애들은 1로 초기화 한다.
  dscores[dscores > 0] = 1
  # svm은 각각의 클래스에 대한 loss값을 모두 더해야 하므로 loss값이 0이 아닌 값들을 모두 더 한다
  valid_dscores_count = np.sum(dscores, axis=0)

  dscores[y, range(X.shape[1])] -= valid_dscores_count

  # backpropate the gradient to the parameters (W, b)
  # scores를 구할 때 W와 X를 곱했었기 때문에 dW를 구할 때는 dscores에 X.T를 곱하여 W가 나오도록 한다.
  dW = np.dot(dscores, X.T)
  db = np.sum(dscores, axis=1, keepdims=True)

  dW += reg * W  # regularization gradient

  return loss, dW, db

def get_gradient_softmax(X, y, W, b, reg):
  """
  fully-vectorized implementation :
  - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
  - y is array of integers specifying correct class (e.g. 50,000-D array)
  - W are weights (e.g. 10 x 3073)
  """

  # -----------------get loss-----------------#

  # X는 전체 5만개의 트레이닝 셋, y는 50000개의 레이블 배열, W는 가중치

  # W와 X를 내적하여 scores 행렬 구함. scores의 dim = 10 x 50000
  scores = W.dot(X) + b
  #print("scores.shape : {}".format(scores.shape))

  exp_scores = np.exp(scores)
  # print("exp_scores.shape : {}".format(exp_scores.shape))
  # 10 x 50000


  sum_exp_scores = np.exp(scores).sum(axis=0, keepdims=True)
  # print("sum_exp_scores.shape : {}".format(sum_exp_scores.shape))
  # exp_scores : 10 x 50000
  # sum_exp_score : 1 x 50000

  softmax = exp_scores / sum_exp_scores
  # print("softmax.shape : {}".format(softmax.shape))

  # 정답 클래스의 값에 -log를 취한다.
  loss_i = -np.log(softmax[y, range(X.shape[1])])

  # 평균 data loss
  data_loss = np.sum(loss_i) / X.shape[1]

  # regularization loss
  reg_loss = 0.5 * reg * np.sum(W * W)

  # 최종 loss
  loss = data_loss + reg_loss

  #-----------------get gradient-----------------#

  # softmax에는 모든 클래스에 대한 확률이 저장되어 있다. loss를 이제 미분한다.
  dscores = softmax
  # 정답 클래스의 확률에서만 1을 빼준다.
  dscores[y, range(X.shape[1])] -= 1
  dscores /= X.shape[1]

  # backpropate the gradient to the parameters (W, b)
  dW = np.dot(dscores, X.T)
  db = np.sum(dscores, axis=1, keepdims=True)

  dW += reg * W  # regularization gradient

  return loss, dW, db

