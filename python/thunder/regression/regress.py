from thunder.utils.common import pinv
from thunder.rdds.series import Series
from numpy import dot, mean, sum, asarray, concatenate


class LocalRegressionModel(object):

    def __init__(self, betas=None):
        self.betas = asarray(betas)

    def fit(self, Xhat, y):
        self.betas = dot(y,Xhat.T)
        return self

    def predict(self, X):
        return dot(self.betas, X)

    def stats(self, X, y):
        yhat = self.predict(X)
        residuals = y - yhat
        const_residuals = y - mean(y)
        ss_resid = sum(residuals**2)
        ss_total = sum(const_residuals**2)
        if ss_total == 0:
            r_squared = 0
        else:
            r_squared = 1 - ss_resid/ss_total
        return asarray([yhat, residuals, r_squared])

    def fit_with_stats(self, Xhat, X, y):
        self.fit(Xhat, y)
        stats = self.stats(X, y)
        return self, stats


class Regression(object):

    def __init__(self):
        pass

    def fit(self, X, y):
        Xhat = pinv(X)
        fitted = y.rdd.map(lambda (k, v): (k, LocalRegressionModel().fit(Xhat, v)))
        return RegressionModel(fitted)

    def fitWithStats(self, X, y):
        Xhat = pinv(X)
        result = y.rdd.map(lambda (k, v): (k, LocalRegressionModel().fit_with_stats(Xhat, X, v)))
        model = RegressionModel(result.map(lambda (k, v): (k, v[0])))
        # TODO: once DataTable is added, then the betas can also be added to the output of stats
        #stats = Series(result.map(lambda (k, v): (k, concatenate((v[0].betas, v[1])))), index=['betas','prediction','residuals','rsquared'])
        stats = Series(result.map(lambda (k, v): (k, v[1])), index=['prediction','residuals','rsquared'])
        return model, stats

class RegressionModel(object):

    def __init__(self, rdd):
        self.rdd = rdd

    @classmethod
    def createModel(cls, series):
        self.rdd = series.rdd.map(lambda (k, v): (k, LocalRegressionModel(v)))
        return self

    def predict(self, X):
        return Series(self.rdd.map(lambda (k, v): (k, v.predict(X))))

    def predictWithStats(self, X, y):
        joined = self.rdd.join(y.rdd)
        result = joined.map(lambda (k, v): (k, v[0].stats(X, v[1])))
        return Series(result, index=['prediction','residuals','rsquared'])

    def getParameters(self):
        return Series(self.rdd.map(lambda (k, v): (k, v.betas)))
