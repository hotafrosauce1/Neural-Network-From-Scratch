import numpy as np
import keras
from keras.utils import np_utils
from sklearn.preprocessing import normalize

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    sig = sigmoid(z)
    return sig * (1 - sig)

class Network:

    def __init__(self, layers):
        self.num_layers = len(layers)
        self.bias = []
        self.weights = []
        self.layers = layers
        self.num_classes = layers[-1]
        self.class_map = {}
        self.inverse_map = {}

        prev_layer_neurons = None
        for (layer_index, neurons) in enumerate(layers):
            if layer_index == 0:
                self.weights.append(np.ones(neurons))
                self.bias.append(np.zeros(neurons))
                prev_layer_neurons = neurons
            else:
                self.weights.append(np.array([np.random.randn(prev_layer_neurons) for _ in range(neurons)]))
                self.bias.append(np.random.rand(neurons))
                prev_layer_neurons = neurons

        self.weights = np.array(self.weights)
        self.bias = np.array(self.bias)

    def predict(self, x):
        output_activation = self.feedforward(x, is_prediction = True)
        prediction_index = np.argmax(output_activation)
        return self.class_map[prediction_index], output_activation

    def train(self, X, y, lr = 0.01, batch_size = 100, epochs = 100):
        #w = w - lr/m * sum{from 1 to m} ()
        assert len(X[0]) == self.layers[0]

        self.create_class_mapping(y)

        X_len = len(X)
        true_batch = min(batch_size, X_len)
        if true_batch < 100:
            batch_size = true_batch

        for epoch in range(epochs):
            X_and_y = np.c_[X, y]

            np.random.shuffle(X_and_y)

            training_batch_X = X_and_y[:batch_size,:-1]
            training_batch_y = X_and_y[:batch_size,-1]

            mapped_outputs = np.array(list(map(lambda x: self.inverse_map[x], training_batch_y)))
            mapped_y = np_utils.to_categorical(mapped_outputs, num_classes = self.num_classes)

            error = 0
            for (x_train_example, y_train_example) in zip(training_batch_X, mapped_y):
                activations, weighted_inputs = self.feedforward(x_train_example)
                errors = self.backpropagation(y_train_example, activations, weighted_inputs)
                self.update_weights(activations, errors, lr, batch_size)

                error += (np.linalg.norm((activations[-1] - y_train_example))) ** 2

            print("Epoch: {}, Error: {}".format(epoch + 1, error))

    def feedforward(self, input, is_prediction = False):
        activations = []
        weighted_inputs = []
        for (layer_index, neurons) in enumerate(self.layers):
            if layer_index == 0:
                weighted_input = self.weights[0] * input
                activations.append(sigmoid(weighted_input))
                weighted_inputs.append(weighted_input)
            else:
                weighted_input = np.dot(self.weights[layer_index], activations[layer_index - 1]) + self.bias[layer_index]
                activations.append(sigmoid(weighted_input))
                weighted_inputs.append(weighted_input)

        activations = np.array(activations)
        weighted_inputs = np.array(weighted_inputs)

        if is_prediction:
            return activations[-1]
        else:
            return activations, weighted_inputs

    def backpropagation(self, y, activations, weighted_inputs):
        errors = {}
        num_layers = len(self.layers)
        for layer_index in range(num_layers - 1, 0, -1):
            if layer_index == num_layers - 1:
                gradient_cost = activations[layer_index] - y
                errors[layer_index] = gradient_cost * sigmoid_prime(weighted_inputs[layer_index])
            else:
                next_layer_weights_t = self.weights[layer_index + 1].T
                next_layer_errors = errors[layer_index + 1]
                curr_layer_weighted_input = weighted_inputs[layer_index]
                curr_layer_error = np.dot(next_layer_weights_t, next_layer_errors) * sigmoid_prime(curr_layer_weighted_input)
                errors[layer_index] = curr_layer_error
        return errors

    def update_weights(self, activations, errors, lr, batch_size):
        for layer_index in range(1, len(self.layers)):
            error_layer = len(errors[layer_index])
            layer_l_minus_1_neuron_count = self.layers[layer_index - 1]

            dC_dw = np.array([[activations[layer_index - 1][row] * errors[layer_index][col]
            for col in range(error_layer)]
            for row in range(layer_l_minus_1_neuron_count)])

            dC_db = errors[layer_index]

            self.weights[layer_index] -= (lr / batch_size * dC_dw.T)
            self.bias[layer_index] -= (lr / batch_size * dC_db)

    def create_class_mapping(self, y):
        class_map = {}
        inverse_map = {}
        curr_class_num = 0
        for val in np.unique(y):
            if val in class_map.keys() and val in inverse_map.values():
                continue
            else:
                class_map[curr_class_num] = val
                inverse_map[val] = curr_class_num
                curr_class_num += 1
        self.class_map = class_map
        self.inverse_map = inverse_map

n = Network([784,30,10,10])
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = list(map(lambda x: x.reshape((784,)), X_train))
X_train = normalize(X_train)

X_test = list(map(lambda x: x.reshape((784,)), X_test))
X_test = normalize(X_test)

n.train(X_train, y_train, batch_size = 100, epochs = 10, lr = 0.01)
num_correct = 0
for i in range(len(X_test)):
    pred, activation = n.predict(X_test[i])
    print("actual: ", y_test[i], "predicted: ", pred, "\n","activation: ", activation)
    if pred == y_test[i]:
        num_correct += 1
accuracy = num_correct / len(X_test)
print("accuracy: {}%".format(accuracy * 100))
