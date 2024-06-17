import math
import bisect
import numpy as np
from scipy.integrate import quad
import torch.nn as nn

# PieceWiseActivation
# This class implements an activation function using piecewise linear approximation.
class PieceWiseActivation(nn.Module):

    def __init__(self):
        super().__init__()
        self.piecewise = PieceWiseApproximation()

    def forward(self, input):
        # Move the input tensor to the CPU, apply operation on CPU, move back to GPU
        input_cpu = input.cpu()
        output_cpu = input_cpu.detach().apply_(self.piecewise.calculate)
        output = output_cpu.to(input.device)
        return output
    
# PieceWiseApproximation
# This class takes GELU, a non-linear activation function, and approximates it with a series of linear segments. 
# The approximation is performed over a specified range of input values.
class PieceWiseApproximation:

    def __init__(self, left=None, right=None, bulk=None, step=0.01, threshold=0.000001):
        if left is not None and right is not None:
            self.create_linedict(left,right,bulk)
        else:
            self.initialize_piecewise(step, threshold)

    def create_linedict(self, left, right, bulk):
        if bulk:
            self.includedPoints = bulk
            self.lineDict = {}
            for x in range(0, len(bulk) - 1):
                self.lineDict[bulk[x]] = self.slopeIntercept(bulk[x], bulk[x + 1])
        else:
            self.includedPoints = [left, right]
            self.lineDict = {left: self.slopeIntercept(left, right)}

    def slopeIntercept(self, xL, xR):
        yR = self.myGelu(xR)
        yL = self.myGelu(xL)
        slope = (yR - yL) / (xR - xL)
        intercept = yR - slope * xR
        return (slope, intercept)

    def calculate(self, x):
        leftBound = bisect.bisect_left(self.includedPoints, x) - 1

        if x <= -10:
            return 0
        elif x >= 10:
            return x

        if leftBound == -1:
            leftBound += 1

        equation = self.lineDict[self.includedPoints[leftBound]]
        return x * equation[0] + equation[1]

    def modify(self, new):
        insert = bisect.bisect_left(self.includedPoints, new)
        self.includedPoints.insert(insert, new)
        self.lineDict[self.includedPoints[insert - 1]] = self.slopeIntercept(self.includedPoints[insert - 1], new)
        self.lineDict[new] = self.slopeIntercept(new, self.includedPoints[insert + 1])

    def initialize_piecewise(self, step=0.01, threshold=0.000001):
        old = -10
        temp = old + step
        bigPoints = [-10]
        last = old

        while temp < 10:
            P = [old, self.myGelu(old)]
            Q = [temp, self.myGelu(temp)]
            chord = math.dist(P, Q)
            arc, arcerror = self.arcLength(old, temp)
            error = abs(arc - chord)

            if error > threshold:
                bigPoints.append(last)
                old = last
                last = temp
            else:
                last = temp
                temp += step

        bigPoints.append(10)
        x = 1

        while x < (len(bigPoints) - 1):
            old, new = bigPoints[x - 1], bigPoints[x + 1]
            P = [old, self.myGelu(old)]
            Q = [new, self.myGelu(new)]
            chord = math.dist(P, Q)
            arc = self.arcLength(old, new)[0]

            if abs(arc - chord) < threshold:
                bigPoints.pop(x)
            else:
                x += 1

        self.create_linedict(-10, 10, bigPoints)

    def unIntCurveLength(self, x):
        return math.sqrt(1 + self.otherPrime(x)**2)

    def arcLength(self, left, right):
        return quad(self.unIntCurveLength, left, right)

    def myGelu(self, x):
        return (0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * pow(x, 3)))))

    def geluPrime(self, x):
        return 0.5 * math.tanh(0.0356774 * math.pow(x, 3) + 0.797885 * x) + (0.0535161 * math.pow(x, 3) + 0.398942 * x) * ((1 / (math.cosh(0.0356774 * math.pow(x, 3) + 0.797885 * x))) ** 2) + 0.5

    def otherPrime(self, x):
        return 0.5 * x * (1 - math.tanh(0.0356774081363001 * x ** 3 + 0.797884560802865 * x) ** 2) * (0.1070322244089 * x ** 2 + 0.797884560802865) + 0.5 * math.tanh(0.0356774081363001 * x ** 3 + 0.797884560802865 * x) + 0.5
