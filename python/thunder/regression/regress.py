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


from numpy import dot, hstack, vstack, zeros, sqrt, ones, eye, array
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
			'linear': LinearRegressionAlgorithm,
			'ridge': RidgeRegressionAlgorithm,
			'tikhonov': RidgeRegressionAlgorithm,
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

	def preprocessing(self):
		raise notImplementedError

	def fit(self, X, y):
		if self.intercept:
			X = hstack([ones([X.shape[0], 1]), X])
			print X
		algorithm, y = self.preprocessing(X, y)
		newrdd = y.rdd.mapValues(lambda v: LocalRegressionModel().fit(algorithm, v))
		return RegressionModel(newrdd)


class LinearRegressionAlgorithm(RegressionAlgorithm):
	'''
	Class for fitting simple linear regression models
	'''

	def __init__(self, **kwargs):
		super(self.__class__, self).__init__(**kwargs)

	def preprocessing(self, X, y):
		algorithm = PseudoInv(X)
		return algorithm, y


class RidgeRegressionAlgorithm(RegressionAlgorithm):
	'''
	Class for fitting ridge regression models
	'''

	def __init__(self, **kwargs):
		super(self.__class__, self).__init__(*kwargs)
		self.c = kwargs['c']

	def preprocessing(self, X, y):
		R = self.c * eye(X.shape[1])
		y = y.applyValues(lambda v: hstack([v, zeros(self.X.shape[1])]))
		algorithm = PsuedoInv(X)
		return algorithm, y


class TikhonovRegressionAlgorithm(RegressionAlgorithm):
	'''
	Class for fitting ridge regression models
	'''

	def __init__(self, **kwargs):
		super(self.__class__, self).__init__(**kwargs)
		self.R = kwargs['R']
		self.c = kwargs['c']

	def preprocessing(self, X, y):
		X = vstack([X, sqrt(self.c)*self.R])
		y  = y.applyValues(lambda v: hstack([v, zeros(self.R.shape[0])]))
		algorithm = PseudoInv(X)
		return algorithm, y


class ConstrainedRegressionAlgorithm(RegressionAlgorithm):
	'''
	Class for fitting regression models with constrains on coefficients
	'''

	def __init__(self, **kwargs):
		super(self.__class__, self).__init__(**kwargs)
		self.A = kwargs['A']
		self.b = kwargs['b']

	def preprocessing(self, X, y):
		y = y.applyValues(lambda v: array(dot(X.T, v), ndmin=2).T)
		P = dot(X.T, X)
		algorithm = QuadProg(P, self.A, self.b)
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

class QuadProg(RegressionFitter):
	'''
	Class for fitting regression models via quadratic programming

	cvxopt.solvers.qp minimizes (1/2)*x'*P*x + q'*x with the constraint Ax <= b
	'''

	def __init__(self, P, A, b):
		self.P = cvxoptMatrix(P)
		self.A = cvxoptMatrix(A)
		self.b = cvxoptMatrix(b)

	def fit(self, q):
		from cvxopt.solvers import qp, options
		options['show_progress'] = False
		q = cvxoptMatrix(q)
		return array(qp(self.P, q, self.A, self.b)['x'])

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

#---------

def cvxoptMatrix(x):
	from cvxopt import matrix
	return matrix(x, x.shape, 'd')

# -------------------------------------------------------------------------------------------