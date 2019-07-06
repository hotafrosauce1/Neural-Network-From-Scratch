import NeuralNetwork
import numpy as np

class NeuralNetworkTest:

    net = NeuralNetwork.Network([3,3,2,10])

    def test_dimensions(self):
        for layer_index, neuron_count in enumerate(self.net.layers):
            if layer_index == 0:
                assert len(self.net.bias[layer_index]) == neuron_count

                num_rows = neuron_count
                expected_weight_dim = (num_rows, )
                actual_weight_dim = self.net.weights[layer_index].shape
                assert expected_weight_dim == actual_weight_dim, "expected: {}, actual: {}".format(expected_weight_dim, actual_weight_dim)

                expected_bias_dim = (neuron_count, 1)
                actual_bias_dim = self.net.bias[layer_index].shape
                assert expected_bias_dim == actual_bias_dim, "expected: {}, actual: {}".format(expected_bias_dim, actual_bias_dim)

            else:
                assert len(self.net.bias[layer_index]) == neuron_count

                num_rows = neuron_count
                num_cols = self.net.layers[layer_index - 1]
                expected_weight_dim = (num_rows, num_cols)
                actual_weight_dim = self.net.weights[layer_index].shape
                assert actual_weight_dim == expected_weight_dim, "expected: {}, actual: {}".format(expected_weight_dim, actual_weight_dim)

                expected_bias_dim = (neuron_count, 1)
                actual_bias_dim = self.net.bias[layer_index].shape
                assert expected_bias_dim == actual_bias_dim, "expected: {}, actual: {}".format(expected_bias_dim, actual_bias_dim)

    def test_variable_types(self):
        for layer_index, neuron_count in enumerate(self.net.layers):
            assert type(self.net.weights[layer_index]) is np.ndarray
            assert type(self.net.bias[layer_index]) is np.ndarray





test = NeuralNetworkTest()
test.test_dimensions()
test.test_variable_types()
