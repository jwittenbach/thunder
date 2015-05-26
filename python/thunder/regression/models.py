from numpy import dot, square, sum, mean
from thunder.rdds.series import Series
from thunder.regression.transformations import applyTransforms
from thunder.utils.common import fastJoin

class RegressionModel(object):

    '''
    Class for fitted regression models.

    Backed by an RDD of of key-value paris. The keys are the tuple identifiers from the
    Series of response variables used in the fit. The values are per-record regression
    models that contain the coefficients from the fit, though these models are not directly
    exposed.
    '''

    def __init__(self, rdd, transforms=[], statsrdd=None, fittingMethod=None):
        self.rdd = rdd
        if not (type(transforms) is list or type(transforms) is tuple):
            transforms = [transforms]
        self.transforms = transforms
        self.statsrdd = statsrdd
        self.fittingMethod = fittingMethod

    def __repr__(self):
        lines = []
        lines.append(self.__class__.__name__)
        if self.transforms == []:
            t = 'None'
        else:
            t = ', '.join([str(x.__class__.__name__) for x in self.transforms])
        lines.append('transformations: ' + t)
        lines.append('fit by: ' + str(self.fittingMethod))
        return '\n'.join(lines)

    @property
    def coeffs(self):
        '''
        Series containing the coefficients of the model.
        '''
        if not hasattr(self, 'coefficients'):
            self.coefficients = Series(self.rdd.mapValues(lambda v: v.betas))
        return self.coefficients

    @property
    def stats(self):
        '''
        Series containing the R-squared values from the original fit of the model.
        '''
        return Series(self.statsrdd)

    def predict(self, X):
        '''
        Predicts the responses given a design matrix

        Parameters
        ----------
        X: array
            Design matrix of shape n x k, where n is the number of samples and k is the
            number of regressors. Even if an intercept term was fit, should NOT include
            a column of ones.

        Returns
        -------
        yhat: Series
            Series of predictions (each of length n)
        '''

        X = applyTransforms(X, self.transforms)
        return Series(self.rdd.mapValues(lambda v: v.predict(X)))

    def score(self, X, y):
        '''
        Computes R-squared values for a single design matrix and multiple responses.

        Parameters
        ----------
        X: array
            Design matrix of shape n x k, where n is the number of samples and k is the
            number of regressors. Even if an intercept term was fit, should NOT include
            a column of ones.

        y: Series
            Series of response variables where each record is a vector of length n, where
            n is the number of samples.

        Returns
        -------
        scores: Series
            Series of R-squared values.
        '''
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