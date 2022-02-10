from keras.datasets import cifar10
import k_nearest_neighbor as knn
import numpy as np

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# flatten out all images to be one-dimensional
Xtr_rows = X_train.reshape(X_train.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = X_test.reshape(X_test.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072

Xtr_rows = Xtr_rows[:10000, :]
Xte_rows = Xte_rows[:2000, :]

print(y_train.shape)
print(y_test.shape)

y_train = y_train[:10000, :]
y_test = y_test[:2000, :]

K_nearest_neighbor = knn.KNearestNeighbor() # create a Nearest Neighbor classifier class
K_nearest_neighbor.train(Xtr_rows, y_train) # train the classifier on the training images and labels

Yte_predict = K_nearest_neighbor.predict(Xte_rows, 5) # predict labels on the test images
Yte_predict = Yte_predict.reshape(y_test.shape[0],y_test.shape[1])

# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print("k nearest neighbor 알고리즘을 사용하여 측정한 정확도는 다음과 같습니다. ")
print('accuracy: %f' % ( np.mean(Yte_predict == y_test) ))
