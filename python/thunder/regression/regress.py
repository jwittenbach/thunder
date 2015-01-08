class LocalRegressionModel(object):

	def __init__(self, betas=None):
		model.betas = betas

	def fit(self, Xhat, y):
		self.betas = dot(Xhat, y)
		return self

	def predict(self, X):
		return dot(self.betas, X)

	def stats(self, X, y):
		yhat = self.predict(X)
		residuals = y - yhat
		const_residuals = y - mean(y)
		ss_redid = sum(residuals**2)
		ss_total = sum(const_rediduals**2)
		if ss_total == 0:
			r_squared = 0
		else:
			r_squared = 1 - ss_resid/ss_total
		return yhat, residuals, r_squared

	def fit_with_stats(self, Xhat, X, y):
		self.fit(Xhat, y)
		stats = self.stats(X, y)
		return (self, stats)

	def predict_with_stats(self, X, y):
		return self.predict(X), self.stats(X, y)

class Regression(object):

	def __init__(self):
		pass

	def fit(self, X, y):
		Xhat = p_inv(X)
		fitted = y.rdd.map(lambda (k, v): (k, LocalRegressionModel().fit(Xhat,v)))
		return RegressionModel(fitted)

	def fitWithStats(self, X, y):
		Xhat = p_inv(X)
		result = y.rdd.map(lambda (k, v): LocalRegressionModel().fit_with_stats(Xhat, X, v))
		return RegressionModel(result.keys), result.values()

class RegressionModel(object):

	def __init__(self, rdd):
		self.rdd = rdd

	@classmethod
	def createModel(cls, rdd):
		self.rdd = rdd.map(lambda (k, v): (k, cls(v)))
		return self

	def predict(self, X):
		return self.rdd.map(lambda (k, v): (k, v.predict(X)))

	def predictWithStats(self, X, y)
		return self.rdd.map(lambda (k, v): (k, v.pre))
