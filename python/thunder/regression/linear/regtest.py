import numpy as np
from thunder import LinearRegression, Series
from sklearn import linear_model as lm


X = np.array([[-0.4309741,   0.43440693,  0.19946369,  1.40428728],
              [ 0.54587086, -1.1092286,  -0.27258427,  0.35205421],
              [-0.4432777,   0.40580108,  0.20938645,  0.26480389],
              [-0.53239659, -0.90966912, -0.13967252,  1.38274305],
              [ 0.35731376,  0.39878607,  0.07762888,  1.82299252],
              [ 0.36687294, -0.17079843, -0.17765573,  0.87161138],
              [ 0.3017848,   1.36537541,  0.91211512, -0.80570055],
              [-0.72330999,  0.36319617,  0.08986615, -0.7830115 ],
              [ 1.11477831,  0.41631623,  0.11104172, -0.90049209],
              [-1.62162968,  0.46928843,  0.62996118,  1.08668594]])

y1 = np.array([ 4.57058016, -4.06400691,  4.25957933,  2.01583617,  0.34791879,
               -0.9113852, 3.41167194,  5.26059279, -2.35116878,  6.28263909])

y = Series(sc.parallelize([ ((1,), y1) ]))


# comparisson data from MATLAB
b1 = np.array([-3.3795, 2.3662, 1.0144, 0.4859])             #linear, no intercept
b2 = np.array([1.3446, -3.8568, 2.9483, -1.7984, -0.1462])   #linear, with intercept
b3 = np.array([1.1598, -3.1967, 1.8821, 0.4835, -0.0221])    #ridge (c=1), normalized + intercept
b4 = np.array([-3.5977, 2.0694, 1.2690, 0])                  #constraint b3<=0

yh1 = np.array([3.3690, -4.5749, 2.7993, 0.1770, 0.7006, -1.4007, 2.7446, 3.0145, -3.1073, 7.7578])
yh2 = np.array([3.7235, -3.5924, 3.8354, 0.7649, 0.7360, -0.3820, 2.6837, 5.1580, -1.7955, 7.6907])
yh3 = np.array([3.4205, -2.8124, 3.4360, 1.0516, 0.7655, -0.4396, 3.2236, 4.2163, -1.5467, 7.5075])
yh4 = np.array([2.7026, -4.6052, 2.7003, -0.1443, -0.3618, -1.8988, 2.8972, 3.4679, -3.0082, 7.6047])

#ordinary least squares
alg1 = LinearRegression(intercept=False)
model1 = alg1.fit(X, y)
beta1 = model1.coeffs.values().collect()[0]
yhat1 = model1.predict(X).values().collect()[0]
print 'OLS'
print '---'
print 'betas: ', np.mean(np.square(b1-beta1))
print 'yhat: ', np.mean(np.square(yh1-yhat1))
print '\n'

alg2 = LinearRegression()
model2 = alg2.fit(X, y)
beta2 = model2.coeffs.values().collect()[0]
yhat2 = model2.predict(X).values().collect()[0]
print 'OLS + intercept'
print '---------------'
print 'betas: ', np.mean(np.square(b2-beta2))
print 'yhat: ', np.mean(np.square(yh2-yhat2))
print '\n'

#ridge regression
R = np.eye(X.shape[1])
c = 1
alg3 = LinearRegression('ridge', c=1, zscore=True)
model3 = alg3.fit(X, y)
beta3 = model3.coeffs.values().collect()[0]
yhat3 = model3.predict(X).values().collect()[0]
print 'ridge'
print '-----'
print 'betas: ', np.mean(np.square(b3-beta3))
print 'yhat: ', np.mean(np.square(yh3-yhat3))
print '\n'


#constrained regression
C = np.array([[0, 0, 0, 1]])
d = np.array([[0]])

alg4 = LinearRegression('constrained', C=C, d=d, intercept=False)
model4 = alg4.fit(X, y)
beta4 = model4.coeffs.values().collect()[0]
yhat4 = model4.predict(X).values().collect()[0]
print 'constrained'
print '-----------'
print 'betas: ', np.mean(np.square(b4-beta4))
print 'yhat: ', np.mean(np.square(yh4-yhat4))
