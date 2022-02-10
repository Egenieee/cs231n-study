import numpy as np

class KNearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k):
        num_test = X.shape[0]

        Ypred = np.zeros(num_test, dtype= self.y_train.dtype)

        for i in range(num_test):
            ## 하나의 테스트 이미지가 들어왔을 때 모든 트레이닝 이미지를 돌면서 테스트 이미지와의 거리 matrix 계산
            distances = np.sum(np.abs(self.X_train - X[i, :]), axis=1)

            ## 이 중 가장 거리가 최소인 트레이닝 이미지 K개 만큼 뽑기
            ## 여기서 저장되는 것은 트레이닝 레이블
            closest_y = self.y_train[np.argsort(distances)[:k]]

            # 1차원 리스트로 reshape
            one_dim_closest_y = closest_y.flatten()
            #print(one_dim_closest_y)

            # ## K개의 가장 거리가 가까운 트레이닝 레이블 중 제일 많이 나온 트레이닝 레이블이 무엇인지 확인
            pred_label = np.bincount(one_dim_closest_y).argmax()
            #print(pred_label)

            ## 가장 투표를 많이 받은 label이 테스트 이미지의 label이 됨
            Ypred[i] = pred_label

            if i % 1000 == 0:
                print("i : {}".format(i))

        return Ypred