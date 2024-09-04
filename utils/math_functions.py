# math_functions.py
import numpy as np
import math
from scipy.integrate import quad

def exponential(x):
    return np.exp(x)

def reciprocal(x):
    return 1 / x

def reciprocal_prime(x):
    return -1/x**2

def inverse_sqrt(x):
    return 1 / np.sqrt(x)

def inverse_sqrt_prime(x):
    return -0.5 * pow(x, -1.5)

def unIntCurveLength(x):
    return math.sqrt(1 + otherPrime(x)**2)

def arcLength(left, right):
    return quad(unIntCurveLength, left, right)

def gelu(x):
    return (0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * pow(x, 3)))))

def gelu_prime(x):
    return 0.5 * math.tanh(0.0356774 * math.pow(x, 3) + 0.797885 * x) + (0.0535161 * math.pow(x, 3) + 0.398942 * x) * ((1 / (math.cosh(0.0356774 * math.pow(x, 3) + 0.797885 * x))) ** 2) + 0.5

def otherPrime(x):
    return 0.5 * x * (1 - math.tanh(0.0356774081363001 * x ** 3 + 0.797884560802865 * x) ** 2) * (0.1070322244089 * x ** 2 + 0.797884560802865) + 0.5 * math.tanh(0.0356774081363001 * x ** 3 + 0.797884560802865 * x) + 0.5
