import math
from scipy.integrate import quad

class PiecewiseSegmentationCalculator:
    def __init__(self, func, left, right, prime):
        self.func = func
        self.left = left
        self.right = right
        self.Prime = prime
        self.splines = self.assign_segments()

    def unIntCurveLength(self, x):
        return math.sqrt(1 + self.Prime(x) ** 2)

    def arc_length(self, left, right):
        return quad(self.unIntCurveLength, left, right)[0]

    def assign_segments(self):
        old = self.left
        step = 0.001
        threshold = 0.000005
        temp = old + step

        big_points = [self.left]
        last = old

        while temp < self.right:
            P = [old, self.func(old)]
            Q = [temp, self.func(temp)]
            chord = math.dist(P, Q)
            arc = self.arc_length(old, temp)

            error = abs(arc - chord)
            if error > threshold:
                big_points.append(last)
                old = last
                last = temp
            else:
                last = temp
                temp += step

        big_points.append(self.right)
        return big_points