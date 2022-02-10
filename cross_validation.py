from keras.datasets import cifar10
import k_nearest_neighbor as knn
import numpy as np

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# flatten out all images to be one-dimensional
Xtr_rows = X_train.reshape(X_train.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = X_test.reshape(X_test.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072

Xval_rows = Xtr_rows[:1000, :]
Xtr_rows = Xtr_rows[1000:, :]
Yval = y_train[:1000]
y_train = y_train[1000:]

validation_accuracies = []
for k in [1,3,5,10,20,50,100]:
    nn = knn.KNearestNeighbor() # create a Nearest Neighbor classifier class
    nn.train(Xtr_rows, y_train) # train the classifier on the training images and labels

    Yval_predict = nn.predict(Xval_rows, k) # predict labels on the test images
    Yval_predict = Yval_predict.reshape(y_test.shape[0],y_test.shape[1])
    acc = np.mean(Yval_predict == Yval)
    print ('accracy: %f in k: %d' % (acc,k))

    validation_accuracies.append((k, acc))