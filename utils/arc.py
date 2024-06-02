import math
from scipy.integrate import quad
from .gelu import otherPrime

def unIntCurveLength(x):
    return math.sqrt(1 + otherPrime(x)**2)

def arcLength(left, right):
    return quad(unIntCurveLength, left, right)