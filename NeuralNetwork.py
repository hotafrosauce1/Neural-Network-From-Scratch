import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class Network:

    def __init__(self, layers):
        self.num_layers = len(layers)
        self.bias = []
        self.weights = []
        prev_layer_neurons = None
        for (layer_index, neurons) in enumerate(layers):
            if layer_index == 0:
                self.weights.append(np.zeros(neurons))
                self.bias.append([])
                prev_layer_neurons = neurons
                continue
            else:
                self.weights.append(np.array([np.zeros(prev_layer_neurons) for _ in range(neurons)]))
                self.bias.append(np.zeros(neurons))
                prev_layer_neurons = neurons

    def train(self, X, y, lr = 0.01, batch_size = 100, epochs = 10):
        return 







n = Network([3,2,3,1])
n.train([np.zeros(5) for _ in range(3)], [1,2,3])

x = np.array([1,2,3])
