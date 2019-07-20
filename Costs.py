class CrossEntropy(object):

    @staticmethod
    def output_error(output_activation, y, weighted_input):
        return output_activation - y

class QuadraticError(object):

    @staticmethod
    def output_error(output_activation, y, weighted_input):
        return (output_activation - y) * sigmoid_prime(weighted_input)
