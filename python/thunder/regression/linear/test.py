class A(object):
    def __new__(cls, val):
        return B(val)
    def __init__(self, val):
        self.val = val

class B(A):
    def __new__(cls, val):
        
    def __init__(self, val):
        super(B, self).__init__(val)
