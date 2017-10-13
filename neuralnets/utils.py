"""Utilities (activation functions)"""
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x >= 0).astype(np.float)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x)**2

def linear(x):  # linear is in fact 'no activation'
    return x

def linear_deriv(x):
    return 1

def softmax(x):  # with numerical stability, as suggested in Karpathy's notes.
    x -= np.max(x, axis=0)
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def softmax_deriv(x):  # don't really care
    raise ValueError('Ooops, should never be called')
