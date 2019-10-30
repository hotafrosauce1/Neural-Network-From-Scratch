# Neural-Network-From-Scratch
# Example Usage of The Network Class
# n = Network([784,100,10])
# (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
# X_train = list(map(lambda x: x.reshape((784,)), X_train))
# X_train = scale(X_train)
#
# X_test = list(map(lambda x: x.reshape((784,)), X_test))
# X_test = scale(X_test)
#
# n.train(X_train, y_train, X_validate = X_test, y_validate = y_test,
#  batch_size = 10, epochs = 10, lr = 0.25, cost_function = 'cross-entropy',
#   regularizaton_method = 'L2', lmbda = 0.1)
