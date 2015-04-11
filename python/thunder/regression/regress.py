from numpy import dot, hstack, vstack, zeros, sqrt
from scipy.linalg import inv

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
			'ridge': RidgeRegressionAlgorithm
		}
		# other options: linear, ridge, lasso, tikhonov, constrained, basis

		return REGALGORITHMS[algorithm](**kwargs)

#---------

class RegressionAlgorithm(object):
	'''
	Abstract base class for all RegressionAlgorithms
	'''

	def __init__(self, **kwargs):
		raise notImplementedError

	def preprocessing(self):
		raise notImplementedError

	def fit(self, X, y):
		algorithm, y = self.preprocessing(X, y)
		newrdd = y.rdd.mapValues(lambda v: LocalRegressionModel().fit(algorithm, v))
		return RegressionModel(newrdd)


class RidgeRegressionAlgorithm(RegressionAlgorithm):
	'''
	Class for fitting ridge regression models
	'''

	def __init__(self, **kwargs):
		self.R = kwargs['R']
		self.c = kwargs['c']

	def preprocessing(self, X, y):
		X = vstack([X, sqrt(self.c)*self.R])
		y  = y.applyValues(lambda v: hstack([v, zeros(self.R.shape[0])]))
		algorithm = PseudoInv(X)
		return algorithm, y

#---------

class RegressionModel(object):
	'''
	Class for fitted regression models
	'''

	def __init__(self, rdd):
		self.rdd = rdd

	def predict(self, X):
		return Series(rdd.mapValues(lambda v: v.predict(X)))

#---------

class RegressionFitter(object):
	'''
	Abstract base class for all regression fitting procedures
	'''

	def __init__(self):
		raise notImplementedError

	def fit(self, y):
		raise notImplementedError

class PseudoInv(RegressionFitter):
	'''
	Class for fitting regression models via a psuedo-inverse
	'''

	def __init__(self, X):
		self.Xhat = dot(inv(dot(X.T, X)), X.T)

	def fit(self, y):
		return dot(self.Xhat, y)

#---------

class LocalRegressionModel(object):
	'''
	Class for fitting and predicting with regression models for each record
	'''

	def __init__(self):
		self.betas = None

	def fit(self, algorithm, y):
		self.betas = algorithm.fit(y)
		return self

	def predict(self, X):
		return dot(X, self.betas)

# -------------------------------------------------------------------------------------------