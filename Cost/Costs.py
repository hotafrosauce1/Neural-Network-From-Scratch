from Custom_Neural_Network.NeuralNetwork import NeuralNetwork

class CrossEntropy(object):
    @staticmethod
    def output_error(output_activation, y, weighted_input):
        return output_activation - y

class QuadraticError(object):
    @staticmethod
    def output_error(output_activation, y, weighted_input):
        return (output_activation - y) * NeuralNetwork.sigmoid_prime(weighted_input)
