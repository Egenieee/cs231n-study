import numpy as np

def L_i(x, y, W):
  """
  unvectorized version. Compute the multiclass svm loss for a single example (x,y)
  - x is a column vector representing an image (e.g. 3073 x 1 in CIFAR-10)
    with an appended bias dimension in the 3073-rd position (i.e. bias trick)
  - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
  - W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)
  """
  ## X는 입력 이미지 하나, y는 정답 레이블 W는 가중치 행렬

  delta = 1.0 # see notes about delta later in this section
  scores = W.dot(x) # scores becomes of size 10 x 1, the scores for each class 각각의 클래스에 대한 점수
  correct_class_score = scores[y] # 정답 레이블이 3이라면 scores에서의 3번째 점수가 정답 클래스의 점수
  D = W.shape[0] # number of classes, e.g. 10
  loss_i = 0.0

  ## 0~9까지 돌면서 정답 클래스를 제외하고 정답이 아닌 클래스들의 Loss값만 구한다.
  for j in range(D): # iterate over all wrong classes
    if j == y:
      # skip for the true class to only loop over incorrect classes
      continue
    # accumulate loss for the i-th example
    loss_i += max(0, scores[j] - correct_class_score + delta)  #SVM
  return loss_i

def L_i_vectorized(x, y, W):
  """
  A faster half-vectorized implementation. half-vectorized
  refers to the fact that for a single example the implementation contains
  no for loops, but there is still one loop over the examples (outside this function)
  """
  delta = 1.0
  scores = W.dot(x)
  # compute the margins for all classes in one vector operation
  margins = np.maximum(0, scores - scores[y] + delta)
  # on y-th position scores[y] - scores[y] canceled and gave delta. We want
  # to ignore the y-th position and only consider margin on max wrong class
  margins[y] = 0
  loss_i = np.sum(margins)
  return loss_i


def L_SVM(X, y, W, b, reg):
  """
  fully-vectorized implementation :
  - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
  - y is array of integers specifying correct class (e.g. 50,000-D array)
  - W are weights (e.g. 10 x 3073)
  """
  # X는 전체 5만개의 트레이닝 셋, y는 50000개의 레이블 배열, W는 가중치

  # W와 X를 내적하여 scores 행렬 구함. scores의 dim = 10 x 50000
  scores = W.dot(X) + b
  #print("scores.shape : {}".format(scores.shape))

  # scores에서 정답인 레이블의 점수만 가져옴 dim = 1 x 50000
  # scores[y, range(X.shape[0])]의 뜻은 scores 배열에서 정답 클래스인 y번째 행과 0부터 시작해서 49999개의 열에서만 scores를 가져오겠다는 뜻.
  correct_class_scores = scores[y, range(X.shape[1])]

  #print("correct_class_scores.shape : {}".format(correct_class_scores.shape))

  # scores에 더해줘야 하는 delta값. scores의 dim과 같은 차원이다 dim = 10 x 50000
  delta = np.ones(scores.shape)

  # scores배열에서 정답 클래스의 점수를 뺀 뒤 delta값을 더해 margins라는 loss값을 담을 배열을 새로 만든다. dim = 10 x 50000
  margins = np.maximum(0, scores - correct_class_scores + delta )

  # 정답 클래스의 loss는 제외시키기 때문에 0으로 초기화
  margins[y, range(X.shape[1])] = 0

  #print("margin.shape : {}".format(margins.shape))

  ## 각 사진 마다의 loss를 구함
  loss_i = np.sum(margins, axis=0)

  # 평균 data loss
  data_loss = np.sum(loss_i) / X.shape[1]

  # regularization loss 구함
  reg_loss = 0.5 * reg * np.sum(W * W)

  # 최종 loss 구함
  loss = data_loss + reg_loss

  #print("margin.shape : {}".format(loss_i.shape))

  return loss

def L_Softmax(X, y, W, b, reg):
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
  # print("exp_correct_scores.shape : {}".format(exp_scores.shape))
  # print("exp_correct_scores: {}".format(exp_scores))
  # 10 x 50000

  sum_exp_scores = np.exp(scores).sum(axis=0, keepdims=True)
  # print("sum_exp_scores.shape : {}".format(sum_exp_scores.shape))
  # print("sum_exp_scores: {}".format(sum_exp_scores))
  # exp_scores : 10 x 50000
  # sum_exp_score : 1 x 50000
  softmax = exp_scores / sum_exp_scores
  # print("softmax.shape : {}".format(softmax.shape))
  #print("softmax : {}".format(softmax))

  # 정답 클래스의 값에 -log를 취한다.
  loss_i = -np.log(softmax[y, range(X.shape[1])])

  # 평균 data loss
  data_loss = np.sum(loss_i) / X.shape[1]

  # regularization loss
  reg_loss = 0.5 * reg * np.sum(W * W)

  # 최종 loss
  loss = data_loss + reg_loss

  return loss