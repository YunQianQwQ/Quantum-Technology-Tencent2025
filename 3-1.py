import numpy as np

def derivative(f, x):
    dx = 1e-5
    y = f(x)
    res = np.ones_like(x)
    for i in range(len(x)):
        x[i] = x[i] + dx
        dy = f(x) - y
        res[i] = dy / dx
        x[i] = x[i] - dx
    return res

# test
x = np.array([1.0,2.0,3.0])
def func(x):
    return 3 * x[0] * x[0] * x[1] * np.sqrt(x[2])
print(derivative(func,x))