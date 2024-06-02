import math

# Define the Gelu function
def myGelu(x):
  return(0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * pow(x, 3)))))

# Define the derivative of the Gelu function
def geluPrime(x):
    return 0.5 * math.tanh(0.0356774 * math.pow(x, 3) + 0.797885 * x) + (0.0535161 * math.pow(x, 3) + 0.398942 * x) * ((1 / (math.cosh(0.0356774 * math.pow(x, 3) + 0.797885 * x))) ** 2) + 0.5

def otherPrime(x):
    return 0.5 * x * (1 - math.tanh(0.0356774081363001 * x ** 3 + 0.797884560802865 * x) ** 2) * (0.1070322244089 * x ** 2 + 0.797884560802865) + 0.5 * math.tanh(0.0356774081363001 * x ** 3 + 0.797884560802865 * x) + 0.5

