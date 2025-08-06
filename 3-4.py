import numpy as np
import tensorcircuit as tc

K = tc.set_backend("tensorflow")
def f(x):
    c = tc.Circuit(1)
    c.rx(0 , theta = x)
    return K.real(c.expectation_ps(z = [0]))

f_grad = K.grad(f)

x = K.convert_to_tensor(np.random.randn(1))
lr = 0.05
step = 100

for _ in range(step):
    g = f_grad(x)
    x = x - g * lr

print(f"min : theta = {x} f = {f(x)}")