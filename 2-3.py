import numpy as np
import matplotlib.pyplot as plt

sigma = [np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.array([[1,0],[0,-1]])]

def f(typeP,typeQ,theta):
    P = sigma[typeP]
    Q = sigma[typeQ]

    v = (np.cos(theta/2) * np.identity(2) + 1j * np.sin(theta/2) * P) @ np.array([1,0])
    return v.conj().T @ Q @ v

plt.figure(figsize=(7,7))
x = np.arange(-np.pi,np.pi,0.01)

for (P,Q) in [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]:
    y = np.array([f(P,Q,theta).real for theta in x])
    plt.plot(x,y,label = f"(P,Q) = {(P,Q)}")

plt.ylim(-2,2)
plt.xlim(-np.pi,np.pi)
plt.legend(loc = 'upper right',ncol = 3)
plt.show()