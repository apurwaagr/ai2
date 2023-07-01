'''
    TODO: Implement the functions: apply_network, forward, backward, update.
    You should use numpy but please do not use any libraries that are not part of the standard library.

    A network is defined by two parameters:
        weights: a list of numpy matrices, where weights[l][i,j] denotes the
                 weight of the edge from the j-ths neuron in layer l to the
                 i-ths neuron in layer l+1.
                 Note that the lecture defined w_{i,j} the other way around.
        biases:  a list of numpy vectors, where biases[l][i] denotes the
                 bias of the i-ths neuron in layer l+1.

    Examples consist of tuples (input_, target) where:
        input_ is a numpy vector containing the input values
        target is a numpy vector containing the expected output values
'''

import numpy as np


def f(x):
    ''' the activation function '''
    return np.tanh(x)

def df(x):
    ''' the derivative of activation function '''
    return 1 - np.tanh(x)**2

def apply_network(weights, biases, input_):
    ''' Returns the output values of the network given the input input_ '''
    activation = input_
    for i in range(len(weights)):
        activation = f(np.dot(weights[i], activation) + biases[i])
    return activation

def forward(weights, biases, input_):
    ''' Performs the forward propagation algorithm.
        Returns (in_, a)
        Where
            in_: contains the "activations before the activation function f was applied",
                 i.e., a[l+1][i] = f(in_[l][i]).
            a:   is the list of numpy vectors containing the neuron activations,
                 i.e., a[l][i] is the activation of the i-th neuron in layer l.
                 Note that a[0] should correspond to input_.
    '''
    activation = input_
    in_ = []
    a = [activation]
    for i in range(len(weights)):
        in_.append(np.dot(weights[i], activation) + biases[i])
        a.append(f(in_[-1]))
        activation = a[-1]
    return in_, a

def backward(in_, a, weights, target):
    ''' Performs the back propagation algorithm.
        `in_` and `a` were computed by forward propagation (`forward`)
        and `target` describes the desired value.

        Returns deltas (Δ), where deltas[l][i] corresponds
        to the i-ths neuron in layer l+1.

        Tip:
            if n is the number of layers - 1:
                deltas[n] = df(in_[n]) * (a[n+1] - target)
            else:
                deltas[n][j] = df(in_[n][j]) * Σ_i weights[n+1][i,j] * deltas[n+1][i]
    '''
    deltas = [None] * len(weights)
    print(len(weights), len(in_), len(a))
    deltas[-1] = df(in_[-1]) * (a[-1] - target)
    for i in range(len(weights)-2,-1,-1):
        deltas[i] = df(in_[i]) * np.dot(weights[i+1].T, deltas[i+1])
    return deltas

def update(weights, biases, delta, a, alpha):
    ''' Returns the updated weights and biases as a tuple (weights, biases).
        `delta` and `a` were computed by the previous functions.
        `alpha` is the step size, following the lecture it is a negative value
        because we want to do gradient *descent*.

        Hints:
            - To use numpy operations, you may have to use `reshape` and `T` (transpose).
            - Don't forget that `weights` has the indices the other way around as in the lecture.
    '''
    for i in range(len(weights)):
        product = np.dot(a[i].reshape((a[i].shape[0], 1)), delta[i].reshape((delta[i].shape[0], 1)).T).T
        weights[i] += alpha * product
        biases[i] += alpha * delta[i]

    return weights, biases


def train(shape, examples, alpha):
    ''' Initializes and trains a neural network.
        Arguments:
            shape:    a list of layer sizes. E.g. [2,3,1] would be a network with 2 input neurons,
                      a hidden layer consisting of 3 neurons and a single neuron in the output layer.
            examples: a list of training examples (pairs of input_ and target).
            alpha:    the step size (should be negative)
        Returns the weights and biases of the trained network
    '''
    # Randomly initialize weights and biases
    weights = []
    biases = []
    for i in range(len(shape)-1):
        weights.append(np.random.random((shape[i+1], shape[i])) - 0.5)
        biases.append(np.random.random(shape[i+1]) - 0.5)

    # train weights and biases with examples
    for input_, target in examples:
        in_, a = forward(weights, biases, input_)
        deltas = backward(in_, a, weights, target)
        weights, biases = update(weights, biases, deltas, a, alpha)

    return weights, biases

