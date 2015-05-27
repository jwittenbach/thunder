from numpy import dot, vstack, sqrt, eye
from thunder.regression.estimators import PseudoInv, TikhonovPseudoInv, QuadProg
from thunder.regression.linear.models import RegressionModel, LocalRegressionModel
from thunder.regression.transformations import AddConstant, ZScore, applyTransforms


class LinearRegression(object):
    """
    Factory class for instantiating mass regression algorithms.

    Parameters
    ----------
    algorithm: string, optional, default = 'linear'
        A string indicating the type of regression algorithm to create. Options are: 'linear',
        'tikhonov', 'ridge', or 'constrained'.

    See also
    --------
    RegressionAlgorithm: Base class for all regression algorithms
    LinearRegressionAlgorithm: Ordinary least squares regression
    TikhonovRegressionAlgorithm: L2 regularization with arbitrary regularization matrix
    RidgeRegressionAlgorithm: L2 regularization with identity regularization matrix
    """
    def __new__(cls, algorithm='ordinary', **kwargs):

        REGALGORITHMS = {
            'ordinary': OrdinaryLinearRegression,
            'tikhonov': TikhonovLinearRegression,
            'ridge': RidgeLinearRegression,
            'constrained': ConstrainedLinearRegression
        }
        # other options: lasso, basis

        return REGALGORITHMS[algorithm](**kwargs)


class LinearRegressionAlgorithm(object):
    """
    Base class for all regression algorithms.

    Parameters
    ----------
    intercept: bool, optional, default = True
        Indicates whether or not a constant intercept term will be included

    zscore: bool, optional, default = False
        Indicates whether or not the data will be zscored (subtract mean and divide by standard deviation so
        that units are standard deviations from the mean) before fitting the model.
    """

    def __init__(self, intercept=True, zscore=False, **extra):
        self.intercept = intercept
        self.zscore = zscore

    def __repr__(self):
        from subprocess import check_output
        className = self.__class__.__name__
        try:
            text = check_output(['cowthink', className])
        except:
            text = className  
        return text

    def _prepare(self, X):
        raise NotImplementedError

    def fit(self, X, y):
        """
        Fit multiple regression models that all use the same design matrix simultaneously.

        Uses a single design matrix and multiple response vectors and estimates a linear regression
        model for each response vector.

        Parameters
        ----------
        X: array
            Common design matrix for all regression models. Shape n x k; n = number of samples, k =
            number of regressors

        y: Series (or a subclass)
            Series of response variables. Each record should be an array of size n.

        Returns
        -------
        model: RegressionModel
            Thunder object for the fitted regression model. Stores the coefficients and can be used
            to make predictions.
        """
        if self.zscore:
            zscore = ZScore(X)
            X = zscore.transform(X)

        estimator, transforms = self._prepare(X)
        newrdd = y.rdd.mapValues(lambda v: LocalRegressionModel().fit(estimator, v))

        if self.intercept:
            transforms.insert(0, AddConstant())

        if self.zscore:
            transforms.insert(0, zscore)
        
        transformedX = applyTransforms(X, transforms)
        stats = newrdd.zip(y.rdd.values()).map(lambda (v1, v2): (v1[0], v1[1].stats(transformedX, v2)))

        return RegressionModel(newrdd, transforms, stats, self.__class__.__name__)


class OrdinaryLinearRegression(LinearRegressionAlgorithm):
    """
    Class for fitting standard linear regression models.

    Uses the psueudo-inverse to compute the OLS estimate for the coefficients where the L2
    norm of the errors are minimized: min over b of (y-Xb)^2

    Parameters
    ----------
    intercept: bool, optional, default = True
        Indicates whether or not a constant intercept term will be included

    normalize: bool, optional, default = False
        Indicates whether or not the data will be normalized (subtract mean and divide by standard deviation so
        that units are standard deviations from the mean) before fitting the model.
    """

    def __init__(self, **kwargs):
        super(OrdinaryLinearRegression, self).__init__(**kwargs)

    def _prepare(self, X):
        if self.intercept:
            X = AddConstant().transform(X)
        estimator = PseudoInv(X)
        transforms = []
        return estimator, transforms


class TikhonovLinearRegression(LinearRegressionAlgorithm):
    """
    Class for fitting Tikhonov regularization regression models.

    Regularizes under-constrained regression problems by penalizing the L2 norm of a vector of
    linear combinations of the coefficients. These linear combinations are specified by a
    regularization matrix 'R' and the amount of regularization by a scalar 'c':
    min over b of (y-Xb)^2 + c(Rb)^2.

    If included, the intercept term is first estimated independently of the regression coefficients
    (as the mean of the response variables) so as to not be included in the regularization.

    Parameters
    ----------
    intercept: bool, optional, default = True
        Indicates whether or not a constant intercept term will be included

    normalize: bool, optional, default = False
        Indicates whether or not the data will be normalized (subtract mean and divide by standard deviation so
        that units are standard deviations from the mean) before fitting the model.

    R: array
        Matrix of size l x k, where l is the number of desired terms in the regularization and k is the
        number of regressors (not including an intercept) in the design matrix.

    c: numeric
        Regularization strength.
    """
    def __init__(self, **kwargs):
        super(TikhonovLinearRegression, self).__init__(**kwargs)
        self.R = kwargs['R']
        self.c = kwargs['c']
        self.nPenalties = self.R.shape[0]

    def _prepare(self, X):
        X = vstack([X, sqrt(self.c) * self.R])
        estimator = TikhonovPseudoInv(X, self.nPenalties, intercept=self.intercept)
        transforms = []
        return estimator, transforms


class RidgeLinearRegression(TikhonovLinearRegression):
    """
    Class for fitting ridge regression models.

    Regularizes under-constrained regression problems by penalizing the L2 norm of the vector
    of coefficients. The amount of regularization is specified by a scalar 'c':
    min over b of (y-Xb)^2 + cb^2. Equivalent to Tikhonov regularization with the identity
    matrix as the regularization matrix.

    If included, the intercept term is first estimated independently of the regression coefficients
    (as the mean of the response variables) so as to not be included in the regularization.

    Parameters
    ----------
    intercept: bool, optional, default = True
        Indicates whether or not a constant intercept term will be included

    normalize: bool, optional, default = False
        Indicates whether or not the data will be normalized (subtract mean and divide by standard deviation so
        that units are standard deviations from the mean) before fitting the model.

    c: numeric
        Regularization strength.
    """
    def __init__(self, **kwargs):
        super(TikhonovLinearRegression, self).__init__(**kwargs)
        self.c = kwargs['c']

    def _prepare(self, X):
        self.nPenalties = X.shape[1]
        self.R = eye(self.nPenalties)
        return super(RidgeLinearRegression, self)._prepare(X)

class ConstrainedLinearRegression(LinearRegressionAlgorithm):
    """
    Class for fitting regression models with constrains on coefficients.

    Given a set of l inequalities that are linear in the regression coefficients, solves the OLS
    problem subject to the constraints imposed by the inequalities via quadratic programming:
    min over b of (y-Xb), given Cb <= d. Here, the matrix C and the vector d specify the linear constraints.

    If included, the intercept term is treated as the first regressor.

    Parameters
    ----------
    intercept: bool, optional, default = True
        Indicates whether or not a constant intercept term will be included

    normalize: bool, optional, default = False
        Indicates whether or not the data will be normalized (subtract mean and divide by standard deviation so
        that units are standard deviations from the mean) before fitting the model.

    C: array
        Matrix of size l x k that specifies the linear combinations of the regression coefficients
        on the LHS of the constraints, where l is the number of constraints and k is the number of
        regressors.

    d: array
        Vector of length l that specifies the threshold values on the RHS of the constraints, where
        l is the number of constraints.
    """
    def __init__(self, **kwargs):
        super(ConstrainedLinearRegression, self).__init__(**kwargs)
        self.C = kwargs['C']
        self.d = kwargs['d']

    def _prepare(self, X):
        if self.intercept:
            X = AddConstant().transform(X)
        estimator = QuadProg(X, self.C, self.d)
        transforms = []
        return estimator, transforms
