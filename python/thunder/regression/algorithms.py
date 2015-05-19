from numpy import dot, hstack, vstack, zeros, sqrt, ones, eye, array, append, mean, std, insert, concatenate, sum, square, eye
from scipy.linalg import inv
from thunder.rdds.series import Series
from thunder.regression.estimators import PseudoInv, TikhonovPseudoInv, QuadProg
from thunder.utils.common import fastJoin, cvxoptMatrix
from thunder.regression.models import RegressionModel, LocalRegressionModel
from thunder.regression.transformations import AddConstant, Scale

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

        estimator, transforms = self.prepare(X)
        newrdd = y.rdd.mapValues(lambda v: LocalRegressionModel().fit(estimator, v))

        if self.intercept or self.normalize:
            transforms.append(AddConstant(X))
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

    def fitWithStats(self, X, y):

        regModel = self.fit(X, y)
        stats = regModel.stats(X, y)
        return regModel, stats


class LinearRegressionAlgorithm(RegressionAlgorithm):

    '''
    Class for fitting simple linear regression models
    '''

    def __init__(self, **kwargs):
        super(LinearRegressionAlgorithm, self).__init__(**kwargs)

    def prepare(self, X):
        if self.intercept:
            X = AddConstant(X).transform(X)
        estimator = PseudoInv(X)
        transforms = []
        return estimator, transforms


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
        estimator = TikhonovPseudoInv(X, self.nPenalties, intercept=self.intercept)
        transforms = []
        return estimator, transforms


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
        estimator = QuadProg(X, self.A, self.b)
        transforms = []
        return estimator, transforms


