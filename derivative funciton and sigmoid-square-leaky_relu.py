# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 09:02:27 2021

@author: leloc
"""

from typing import Callable
from numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt


input_range = np.arange(-5, 5, 0.01)

def square(x: ndarray) -> ndarray:
    '''
    Aplica função quadrada no ndarray.
    '''
    return np.power(x, 2)
def leaky_relu(x: ndarray) -> ndarray:
    '''
    Aplica função "Leaky ReLU" para cada elemento no ndarray.
    '''
    return np.maximum(0.2 * x, x)   

def sigmoid(x: ndarray) -> ndarray:
    '''
    Aplica função "sigmoide" para cada elemento no ndarray
    '''
    return 1 / (1 + np.exp(-x))

def deriv(func: Callable[[ndarray], ndarray],
          input_: ndarray,
          diff: float = 0.001) -> ndarray:
    '''
    Executa a derivada da função "func" para cada elemento no "input_" array.
    '''
    return (func(input_ + diff) - func(input_ - diff)) / (2 * diff)


S = square(input_range)
dSdN = deriv(square, input_range)


#S = sigmoid(input_range)
#dSdN = deriv(sigmoid, input_range)

#S = leaky_relu(input_range)
#dSdN = deriv(leaky_relu, input_range)



plt.grid()
plt.plot(input_range,S)
plt.plot(input_range,dSdN)