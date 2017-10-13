"""neuralnets - A basic NN library"""

from .neural_nets import NeuralNet
from .logistic_reg import LogisticReg
from .two_layers import TwoLayersNN

__version__ = '0.1.0'
__author__ = 'Nicolas Hug <contact@nicolas-hug.com>'
__all__ = ['NeuralNet', 'LogisticReg', 'TwoLayersNN']
