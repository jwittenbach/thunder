from numpy import mean, std, hstack, ones

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

    def __init__(self):
        pass

    def transform(self, X):
        return hstack([ones([X.shape[0], 1]), X])

#---------

def applyTransforms(X, transforms):
    if not (isinstance(transforms, list) or isinstance(transforms, tuple)):
        transforms = [transforms]
    for t in transforms:
        X = t.transform(X)
    return X

