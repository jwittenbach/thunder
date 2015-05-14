from numpy import dot, hstack, vstack, zeros, sqrt, ones, eye, array, append, mean, std, insert, concatenate, sum, square, eye
from scipy.linalg import inv
from thunder.rdds.series import Series


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

        estimator, transforms = self.prepare(X)
        newrdd = y.rdd.mapValues(lambda v: LocalRegressionModel().fit(estimator, v))

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
        X = applyTransforms(X, self.transforms)
        return Series(self.rdd.mapValues(lambda v: v.predict(X)))

    def stats(self, X, y):
        X = applyTransforms(X, self.transforms)
        joined = fastJoin(self.rdd, y.rdd)
        newrdd = joined.mapValues(lambda (model, y): model.stats(X, y))
        return Series(newrdd)

    def predictWithStats(self, X, y):
        X = applyTransforms(X, self.transforms)
        joined = fastJoin(self.rdd, y.rdd)
        results = joined.mapValues(lambda (model, y): model.predictWithStats(X, y))
        yhat = results.mapValues(lambda v: v[0])
        stats = results.mapValues(lambda v: v[1])
        return Series(yhat), Series(stats)

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

    def fit(self, estimator, y):
        self.betas = estimator.fit(y)
        return self

    def predict(self, X):
        return dot(X, self.betas)

    def stats(self, X, y, yhat=None):
        if yhat is None:
            yhat = self.predict(X)

        SST = sum(square(y - mean(y)))
        SSR = sum(square(y - yhat))

        if SST == 0:
            Rsq = 1
        else:
            Rsq = 1 - SSR/SST

        return Rsq

    def predictWithStats(self, X, y):
        yhat = self.predict(X)
        stats = self.stats(X, y, yhat)
        return yhat, stats

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

def applyTransforms(X, transforms):
    if not (type(transforms) is list or type(transforms) is tuple):
        transforms = [transforms]
    for t in transforms:
        X = t(X).transform(X)
    return X

def fastJoin(rdd1, rdd2):
    '''
    function to quickly join two rdds

    assumes that both rdds contain key-value pairs and that they are
    related through a series of maps on the values
    '''

    try:
        return rdd1.zip(rdd2).map(lambda (x, y): (x[0], [x[1], y[1]]))
    except:
        raise ValueError("could not do a fast join on rdds - should be related by a map on values")
