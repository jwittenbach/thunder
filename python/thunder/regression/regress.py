from thunder.utils.common import pinv
from numpy import dot, mean, sum


class LocalRegressionModel(object):

	def __init__(self, betas=None):
		self.betas = betas

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
		return yhat, residuals, r_squared

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
		return RegressionModel(result.map(lambda (k, v): (k, v[0]))), result.map(lambda (k, v): (k, v[1]))

class RegressionModel(object):

	def __init__(self, rdd):
		self.rdd = rdd

	@classmethod
	def createModel(cls, rdd):
		self.rdd = rdd.map(lambda (k, v): (k, cls(v)))
		return self

	def predict(self, X):
		return self.rdd.map(lambda (k, v): (k, v.predict(X)))

	def predictWithStats(self, X, y):
		return self.rdd.map(lambda (k, v): (k, v.predict_with_stats(X,y)))

	def getParameters(self):
		return self.rdd.map(lambda (k, v): (k, v.betas))
