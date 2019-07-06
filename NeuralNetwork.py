import numpy as np
import keras
from keras.utils import np_utils

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
                self.weights.append(np.random.randn(neurons))
                self.bias.append(np.zeros(neurons).reshape((neurons,1)))
                prev_layer_neurons = neurons
                continue
            else:
                self.weights.append(np.array([np.random.randn(prev_layer_neurons) for _ in range(neurons)]))
                self.bias.append(np.random.randn(neurons).reshape((neurons,1)))
                prev_layer_neurons = neurons

        self.weights = np.array(self.weights)
        self.bias = np.array(self.bias)

    def predict(self, x):
        output_activation = self.feedforward(x, is_prediction = True)
        prediction_index = np.argmax(output_activation)
        print(self.class_map)
        print(self.class_map[prediction_index])


    def train(self, X, y, lr = 0.01, batch_size = 100, epochs = 100):
        #w = w - lr/m * sum{from 1 to m} ()
        assert len(X[0]) == self.layers[0]

        self.create_class_mapping(y)

        activations = [np.array([])]
        true_batch = min(batch_size, len(X))
        if true_batch < 100:
            batch_size = true_batch

        for epoch in range(epochs):
            holder = [(X[row], y[row]) for row in range(batch_size)]

            y = np.reshape(y, (len(y), 1))
            X_and_y = np.array(holder)

            np.random.shuffle(X_and_y)

            X = [X_and_y[i][0] for i in range(batch_size)]
            y = [X_and_y[i][1] for i in range(batch_size)]

            training_batch_X = X[:batch_size]
            training_batch_y = y[:batch_size]

            mapped_outputs = np.array(list(map(lambda x: self.inverse_map[x], training_batch_y)))
            mapped_y = np_utils.to_categorical(mapped_outputs, num_classes = self.num_classes)

            error = 0
            for (x_train_example, y_train_example) in zip(training_batch_X, mapped_y):
                activations, weighted_inputs = self.feedforward(x_train_example)
                errors = self.backpropagation(y_train_example, activations, weighted_inputs)
                self.update_weights(activations, errors)

                error += (np.linalg.norm((activations[-1] - y_train_example))) ** 2

            print("Epoch: {}, Error: {}".format(epoch, error))

    def feedforward(self, input, is_prediction = False):
        input = input.reshape((input.shape[0] * input.shape[1], 1))
        activations = []
        weighted_inputs = []
        for (layer_index, neurons) in enumerate(self.layers):
            if layer_index == 0:
                input_weights_shape = self.weights[0].shape
                weighted_input = self.weights[0].reshape((input_weights_shape[0], 1)) * input
                activations.append(sigmoid(weighted_input))
                weighted_inputs.append(weighted_input)
            else:
                weighted_input = np.dot(self.weights[layer_index], activations[layer_index - 1]) + self.bias[layer_index]
                activations.append(sigmoid(weighted_input))
                weighted_inputs.append(weighted_input)

        if is_prediction:
            return activations[-1]
        else:
            return activations, weighted_inputs

    def backpropagation(self, y, activations, weighted_inputs):
        errors = {}
        num_layers = len(self.layers)
        for layer_index in range(num_layers - 1, 0, -1):
            if layer_index == 0:
                return errors
            elif layer_index == num_layers - 1:
                gradient_cost = activations[layer_index] - y
                errors[layer_index] = gradient_cost * sigmoid_prime(weighted_inputs[layer_index])
            else:
                next_layer_weights_t = self.weights[layer_index + 1].T
                next_layer_errors = errors[layer_index + 1]
                curr_layer_weighted_input = weighted_inputs[layer_index]
                curr_layer_error = np.dot(next_layer_weights_t, next_layer_errors) * sigmoid_prime(curr_layer_weighted_input)
                errors[layer_index] = curr_layer_error
        return errors

    def update_weights(self, activations, errors):
        for layer_index in range(1, len(self.layers)):
            # for weight_matrix_row in range(self.layers[layer_index]):
            error_layer = len(errors[layer_index])
            layer_l_minus_1_neuron_count = self.layers[layer_index - 1]
            dC_dw = np.array([[activations[layer_index - 1][row] * errors[layer_index][col]
            for col in range(error_layer)]
            for row in range(layer_l_minus_1_neuron_count)])
            dC_db = errors[layer_index]

            print("layer index: ", layer_index)
            print(self.weights[layer_index].shape, dC_dw.T.shape)

            self.weights[layer_index] -= dC_dw.T
            self.bias[layer_index] -= dC_db


    def create_class_mapping(self, y):
        class_map = {}
        inverse_map = {}
        curr_class_num = 0
        for val in np.unique(y):
            if val in class_map.keys() and val in inverse_map.keys():
                continue
            else:
                class_map[curr_class_num] = val
                inverse_map[val] = curr_class_num
                curr_class_num += 1
        self.class_map = class_map
        self.inverse_map = inverse_map

n = Network([784,1,10])
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = list(map(lambda x: x.reshape((784, 1)), X_train))
n.train(X_train, y_train, batch_size = 300, epochs = 1)
