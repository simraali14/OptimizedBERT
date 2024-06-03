import math
import bisect
import numpy as np
from gelu import myGelu
from arc import arcLength

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
        yR = myGelu(xR)
        yL = myGelu(xL)
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
            P = [old, myGelu(old)]
            Q = [temp, myGelu(temp)]
            chord = math.dist(P, Q)
            arc, arcerror = arcLength(old, temp)
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
            P = [old, myGelu(old)]
            Q = [new, myGelu(new)]
            chord = math.dist(P, Q)
            arc = arcLength(old, new)[0]

            if abs(arc - chord) < threshold:
                bigPoints.pop(x)
            else:
                x += 1

        self.create_linedict(-10, 10, bigPoints)
