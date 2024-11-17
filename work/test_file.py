
import numpy as np
def mnk(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    a,b = np.linalg.lstsq(A, y)[0]
    return a,b
np.random.seed(10)
x = np.linspace(0, 10, 30)
y = np.linspace(0, 10, 30)*(10+4*np.random.rand(30)) + (10+4*np.random.rand(30))
a,b = mnk(x,y)
y_pred = a*5+b
res = np.array([a,b,y_pred])
y_true = 12*5+12
print(np.linalg.norm(res - np.array([ 12.,  12.,  72])) <2.)