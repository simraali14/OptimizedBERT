import bisect

class PiecewiseApproximation:
    """
    A class to create a piecewise linear approximation of a given function over a specified interval.
    
    Attributes:
    -----------
    includedPoints : list
        List of points at which the function is approximated.
    lineDict : dict
        Dictionary that stores the slope and intercept for each segment.
    """

    def __init__(self, left, right, func, bulk=None):
        """
        Initializes the piecewise approximation with given interval and function.

        Parameters:
        -----------
        left : float
            The left bound of the interval.
        right : float
            The right bound of the interval.
        func : function
            The function to approximate.
        bulk : list, optional
            A list of precomputed points for the approximation. If provided,
            these points will be used to create the segments.
        """
        if bulk:
            self.includedPoints = bulk
            self.lineDict = {}
            for x in range(0, len(bulk) - 1):
                self.lineDict[bulk[x]] = self.slopeIntercept(bulk[x], bulk[x + 1], func)
        else:
            self.includedPoints = [left, right]
            self.lineDict = {left: self.slopeIntercept(left, right, func)}

    def slopeIntercept(self, xL, xR, func):
        """
        Computes the slope and intercept for a linear segment between xL and xR.

        Parameters:
        -----------
        xL : float
            The left bound of the segment.
        xR : float
            The right bound of the segment.
        func : function
            The function to approximate.

        Returns:
        --------
        tuple
            A tuple containing the slope and intercept of the segment.
        """
        yL = func(xL)
        yR = func(xR)
        slope = (yR - yL) / (xR - xL)
        intercept = yR - slope * xR
        return slope, intercept

    def calculate(self, x):
        """
        Calculates the approximate value of the function at point x using the piecewise linear approximation.

        Parameters:
        -----------
        x : float
            The point at which to calculate the approximation.

        Returns:
        --------
        float
            The approximate value of the function at point x.
        """
        leftBound = bisect.bisect_left(self.includedPoints, x) - 1
        if leftBound == -1:
            leftBound += 1
        equation = self.lineDict[self.includedPoints[leftBound]]
        return x * equation[0] + equation[1]

    def modify(self, new, func):
        """
        Adds a new point to the piecewise approximation and updates the segments accordingly.

        Parameters:
        -----------
        new : float
            The new point to add to the approximation.
        func : function
            The function to approximate.
        """
        insert = bisect.bisect_left(self.includedPoints, new)
        self.includedPoints.insert(insert, new)
        self.lineDict[self.includedPoints[insert - 1]] = self.slopeIntercept(self.includedPoints[insert - 1], new, func)
        self.lineDict[new] = self.slopeIntercept(new, self.includedPoints[insert + 1], func)
