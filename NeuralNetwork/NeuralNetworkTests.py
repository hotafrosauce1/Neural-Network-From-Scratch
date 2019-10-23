from Custom_Neural_Network.NeuralNetwork import NeuralNetwork
import numpy as np

class NeuralNetworkTest:

    net = NeuralNetwork.Network([3,30,30,3])

    def test_dimensions(self):
        for layer_index, neuron_count in enumerate(self.net.layers):
            if layer_index == 0:
                assert len(self.net.bias[layer_index]) == neuron_count

                num_rows = neuron_count
                expected_weight_dim = (num_rows, )
                actual_weight_dim = self.net.weights[layer_index].shape
                assert expected_weight_dim == actual_weight_dim, "expected: {}, actual: {}".format(expected_weight_dim, actual_weight_dim)

                expected_bias_dim = (neuron_count, )
                actual_bias_dim = self.net.bias[layer_index].shape
                assert expected_bias_dim == actual_bias_dim, "expected: {}, actual: {}".format(expected_bias_dim, actual_bias_dim)

            else:
                assert len(self.net.bias[layer_index]) == neuron_count

                num_rows = neuron_count
                num_cols = self.net.layers[layer_index - 1]
                expected_weight_dim = (num_rows, num_cols)
                actual_weight_dim = self.net.weights[layer_index].shape
                assert actual_weight_dim == expected_weight_dim, "expected: {}, actual: {}".format(expected_weight_dim, actual_weight_dim)

                expected_bias_dim = (neuron_count, )
                actual_bias_dim = self.net.bias[layer_index].shape
                assert expected_bias_dim == actual_bias_dim, "expected: {}, actual: {}".format(expected_bias_dim, actual_bias_dim)

    def test_variable_types(self):
        for layer_index, neuron_count in enumerate(self.net.layers):
            assert type(self.net.weights[layer_index]) is np.ndarray
            assert type(self.net.bias[layer_index]) is np.ndarray

    def test_predictions(self):
        y = []
        X = []
        for _ in range(22):
            X.append([1,2,3])
            y.append(0)

        for _ in range(22):
            X.append([4,5,6])
            y.append(1)

        for _ in range(22):
            X.append([7,8,9])
            y.append(2)

        X = np.array(X)
        y = np.array(y)

        self.net.train(X, y, epochs = 500)

        correct = 0

        Xandy = np.c_[X, y]
        for _ in range(100):
            np.random.shuffle(Xandy)
            x = Xandy[:,:-1][0]
            actual = Xandy[:,-1][0]
            pred, act = self.net.predict(x)
            print("input: ", x, "prediction: ", pred, "actual: ", actual ,"activation output: ", act)
            if actual == pred:
                correct += 1
        print("accuracy: {}%".format(correct))


test = NeuralNetworkTest()
test.test_predictions()
