'''
Example code
------------
import numpy as np
from thunder import Regression, Series

y1, y2 = np.array([1,2,3]), np.array([4,5,6])
y = Series(sc.parallelize([ ((1,), y1), ((2,), y2) ]))

X = np.array([[1, 1], [1, 2], [1, 3]])

R = np.array([[1, 0], [0, 1]])
c = 0

alg = Regression('tikhonov', R=R, c=c)
model = alg.fit(X, y)

print model.rdd.mapValues(lambda v: v.betas).values().collect()
'''


from numpy import dot, hstack, vstack, zeros, sqrt, ones, eye, array, append, mean, std, insert, concatenate, sum, square, eye
from scipy.linalg import inv
from thunder.rdds.series import Series

class RegressionBuilder(object):

    '''
    Factory class for creating specialized GeneralRegressionAlgoritms that allow detailed
    specification of all aspects of the regression.

    Paramters that can be specified:
        (1) optimization technique: psuedo-inverse, quadratic programming, gradient descent
        (2) regularization type: 'lasso', 'ridge', 'tikhonov'
        (3) regularization parameters: tihnohov matrix, regularization strength (aka lambda)
        (4) equality constraints: constraint matrix, RHS vector
        (5) inequality constraints: constraint matrix, RHS vector
        (6) set if basis functions
    '''

    def __new__(cls, **kwargs):
        return CustomRegressionAlgorithm(kwargs)


class Regression(object):

    '''
    Factory class for creating RegressionAlgorithms that implement common patterns

    '''
    def __new__(cls, algorithm='linear', **kwargs):

        REGALGORITHMS = {
            'linear': LinearRegressionAlgorithm,
            'tikhonov': TikhonovRegressionAlgorithm,
            'ridge': RidgeRegressionAlgorithm,
            'constrained': ConstrainedRegressionAlgorithm
        }
        # other options: linear, ridge, lasso, tikhonov, constrained, basis

        return REGALGORITHMS[algorithm](**kwargs)

#---------


class RegressionAlgorithm(object):

    '''
    Abstract base class for all RegressionAlgorithms
    '''

    def __init__(self, **kwargs):
        if kwargs.has_key('intercept') and not kwargs['intercept']:
            self.intercept = False
        else:
            self.intercept = True

        if kwargs.has_key('normalize') and kwargs['normalize']:
            self.normalize = True
        else:
            self.normalize = False

    def prepare(self):
        raise NotImplementedError

    def fit(self, X, y):

        if self.normalize:
            scale = Scale(X)
            X = scale.transform(X)

        algorithm, transforms = self.prepare(X)
        newrdd = y.rdd.mapValues(lambda v: LocalRegressionModel().fit(algorithm, v))

        if self.intercept or self.normalize:
            transforms.append(AddConstant)
            index = 1
        else:
            index = 0

        if self.normalize:
                def unscale(betas, scale):
                    if self.intercept:
                        b0 = betas[0]
                        start = 1
                    else:
                        b0 = 0
                        start = 0
                    slopes = betas[start:] / scale.std
                    intercept = b0 - dot(scale.mean, slopes)
                    return insert(slopes, 0, intercept)
                newrdd = newrdd.mapValues(lambda v: v.setBetas(unscale(v.betas, scale)))

        return RegressionModel(newrdd, transforms)


class LinearRegressionAlgorithm(RegressionAlgorithm):

    '''
    Class for fitting simple linear regression models
    '''

    def __init__(self, **kwargs):
        super(LinearRegressionAlgorithm, self).__init__(**kwargs)

    def prepare(self, X):
        if self.intercept:
            X = AddConstant(X).transform(X)
        algorithm = PseudoInv(X)
        transforms = []
        return algorithm, transforms


class TikhonovRegressionAlgorithm(RegressionAlgorithm):
    '''
    Class for fitting Tikhonov regularization regression models
    '''

    def __init__(self, **kwargs):
        super(TikhonovRegressionAlgorithm, self).__init__(**kwargs)
        self.intercept = True
        self.normalize = True
        self.R = kwargs['R']
        self.c = kwargs['c']
        self.nPenalties = self.R.shape[0]

    def prepare(self, X):
        X = vstack([X, sqrt(self.c) * self.R])
        algorithm = TikhonovPseudoInv(X, self.nPenalties, intercept=self.intercept)
        transforms = []
        return algorithm, transforms


class RidgeRegressionAlgorithm(TikhonovRegressionAlgorithm):
    '''
    Class for fitting ridge regression models
    '''

    def __init__(self, **kwargs):
        n = kwargs['n']
        c = kwargs['c']
        R = eye(n)
        super(RidgeRegressionAlgorithm, self).__init__(R=R, **kwargs)

class ConstrainedRegressionAlgorithm(RegressionAlgorithm):

    '''
    Class for fitting regression models with constrains on coefficients
    '''

    def __init__(self, **kwargs):
        super(ConstrainedRegressionAlgorithm, self).__init__(**kwargs)
        self.A = kwargs['A']
        self.b = kwargs['b']

    def prepare(self, X):
        if self.intercept:
            X = AddConstant(X).transform(X)
        algorithm = QuadProg(X, self.A, self.b)
        transforms = []
        return algorithm, transforms


#---------

class RegressionModel(object):

    '''
    Class for fitted regression models
    '''

    def __init__(self, rdd, transforms=[]):
        self.rdd = rdd
        if not (type(transforms) is list or type(transforms) is tuple):
            transforms = [transforms]
        self.transforms = transforms

    @property
    def coefs(self):
        return Series(self.rdd.mapValues(lambda v: v.betas))

    def predict(self, X):
        X = applyTranforms(X, self.transforms)
        return self.rdd.mapValues(lambda v: v.predict(X))

    def stats(self, X, y):
        X = applyTransforms(X, self.transforms)
        return self.rdd.mapsValues(lambda v: v.score(X, y))

#---------

class RegressionEstimator(object):
    '''
    Abstract base class for all regression fitting procedures
    '''

    def __init__(self, intercept=False):
        self.intercept = intercept

    def estimate(self, y):
        raise NotImplementedError

    def fit(self, y):
        if self.intercept:
            b0 = mean(y)
            y = y - b0

        b = self.estimate(y)

        if self.intercept:
            b = insert(b, 0, b0)
        return b

class PseudoInv(RegressionEstimator):
    '''
    Class for fitting regression models via a psuedo-inverse
    '''

    def __init__(self, X, **kwargs):
        super(PseudoInv, self).__init__(**kwargs)
        self.Xhat = dot(inv(dot(X.T, X)), X.T)

    def estimate(self, y):
        return dot(self.Xhat, y)

class TikhonovPseudoInv(PseudoInv):
    '''
    Class for fitting Tikhonov regularization models via a psuedo-inverse
    '''

    def __init__(self, X, nPenalties, **kwargs):
        self.nPenalties = nPenalties
        super(TikhonovPseudoInv, self).__init__(X, **kwargs)

    def estimate(self, y):
        y = hstack([y, zeros(self.nPenalties)])
        return super(TikhonovPseudoInv, self).estimate(y)


class QuadProg(RegressionEstimator):
    '''
    Class for fitting regression models via quadratic programming

    cvxopt.solvers.qp minimizes (1/2)*x'*P*x + q'*x with the constraint Ax <= b
    '''

    def __init__(self, X, A, b, **kwargs):
        super(QuadProg, self).__init__(**kwargs)
        self.X = X
        self.P = cvxoptMatrix(dot(X.T, X))
        self.A = cvxoptMatrix(A)
        self.b = cvxoptMatrix(b)

    def estimate(self, y):
        from cvxopt.solvers import qp, options
        options['show_progress'] = False
        q = cvxoptMatrix(array(dot(-self.X.T, y), ndmin=2).T)
        return array(qp(self.P, q, self.A, self.b)['x']).flatten()

#------

class LocalRegressionModel(object):
    '''
    Class for fitting and predicting with regression models for each record
    '''

    def __init__(self, betas=None):
        self.betas = betas

    def fit(self, algorithm, y):
        self.betas = algorithm.fit(y)
        return self
    
    def getPrediction(self, X):
        return dot(X, self.betas)

    def getStats(self, y, yhat):
        SST = sum(square(y - mean(y)))
        SSR = sum(square(y - yhat))

        if SST == 0:
            Rsq = 1
        else:
            Rsq = 1 - SSR/SST

        return Rsq

    def predict(self, X):
        return self.getPrediction(X)

    def score(self, X, y):
        yhat = self.getPrediction(X)
        return self.getStats(y, yhat)

    def predictAndScore(self, X, y):
        yhat = self.getPrediction(X)
        return yhat, self.getStats(y, yhat)

    def setBetas(self, betas):
        self.betas = betas
        return self

#---------

class Transformation(object):
    '''
    Class for transforming data before fitting/predicting
    '''

    def __init__(self):
        raise NotImplementedError

    def transform(self):
        raise NotImplementedError

class Scale(Transformation):
    '''
    Class for scaling data
    '''

    def __init__(self, X):
        self.mean = mean(X, axis=0)
        self.std = std(X, ddof=1, axis=0)

    def transform(self, X):
        return (X - self.mean)/self.std

class Center(Transformation):
    '''
    Class for centering data
    '''

    def __init__(self, X):
        self.mean = mean(X, axis=0)

    def transform(self, X):
        return X - self.mean

class AddConstant(Transformation):
    '''
    Class for adding a column of 1s to a data matrix
    '''

    def __init__(self, X):
        pass

    def transform(self, X):
        return hstack([ones([X.shape[0], 1]), X])

#---------

def cvxoptMatrix(x):
    from cvxopt import matrix
    return matrix(x, x.shape, 'd')

def applyTranforms(X, transforms):
    if not (type(transforms) is list or type(transforms) is tuple):
        transforms = [transforms]
    for t in transforms:
        X = t(X).transform(X)
    return X

