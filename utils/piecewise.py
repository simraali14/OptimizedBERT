import math
import bisect
from scipy.integrate import quad

class Piecewise:

    def __init__(self, left, right, func, prime):
        self.left = left
        self.right = right
        self.func = func
        self.prime = prime
        self.splines = self._assign_segments()
        self.segment_data = self._calculate_segment_data()

    def _unintegrated_curve_length(self, x):
        return math.sqrt(1 + self.prime(x) ** 2)

    def _arc_length(self, left, right):
        return quad(self._unintegrated_curve_length, left, right)[0]

    def _assign_segments(self):
        old = self.left
        step = 0.001
        threshold = 0.000005
        temp = old + step

        big_points = [self.left]
        last = old

        while temp < self.right:
            P = [old, self.func(old)]
            Q = [temp, self.func(temp)]
            chord_length = math.dist(P, Q)
            arc_length = self._arc_length(old, temp)

            error = abs(arc_length - chord_length)
            if error > threshold:
                big_points.append(last)
                old = last
                last = temp
            else:
                last = temp
                temp += step

        big_points.append(self.right)
        return big_points

    def _slope_intercept(self, xL, xR):
        yL = self.func(xL)
        yR = self.func(xR)
        slope = (yR - yL) / (xR - xL)
        intercept = yR - slope * xR
        return slope, intercept

    def _calculate_segment_data(self):
        segment_data = {}
        for i in range(len(self.splines) - 1):
            xL = self.splines[i]
            xR = self.splines[i + 1]
            slope, intercept = self._slope_intercept(xL, xR)
            segment_data[xL] = (slope, intercept)
        return segment_data
    
    def calculate(self, x):
        left_bound = bisect.bisect_left(self.splines, x) - 1

        if left_bound < 0: left_bound = 0
        if left_bound > len(self.splines)-2: left_bound = len(self.splines) - 2
        
        segment = self.splines[left_bound]
        equation = self.segment_data[segment]
        return x * equation[0] + equation[1]

    def modify(self, new):
        insert = bisect.bisect_left(self.splines, new)
        self.splines.insert(insert, new)
        slope, intercept = self._slope_intercept(self.splines[insert - 1], new)
        self.segment_data[self.splines[insert - 1]] = (slope, intercept)
        slope, intercept = self._slope_intercept(new, self.splines[insert + 1])
        self.segment_data[new] = (slope, intercept)
