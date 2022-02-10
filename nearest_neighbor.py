import numpy as np

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]

    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # 모든 이미지 마다 실행
    for i in range(num_test):
      ## 하나의 테스트 이미지가 들어왔을 때 모든 트레이닝 이미지를 돌면서 테스트 이미지와의 거리 matrix 계산
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)

      ## 이 중 거리가 가장 최소인 트레이닝 이미지 뽑기
      min_index = np.argmin(distances)

      ## 뽑힌 트레이닝 이미지의 클래스가 테스트 이미지의 클래스로 결정됨
      Ypred[i] = self.ytr[min_index]
      print("i:{}".format(i))

    return Ypred