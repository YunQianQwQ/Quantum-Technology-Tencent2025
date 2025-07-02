import numpy as np

def Exp(A, step = 30):
    res = np.zeros_like(A)
    now = np.identity(A.shape[0])
    for i in range(step):
        res = res + now
        now = np.dot(now,A)/(i+1)
    return res

sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0,-1j],[1j,0]])
sigma_z = np.array([[1,0],[0,-1]])

print("exp(i * sigma_x) =")
print(Exp(1j*sigma_x))
print("exp(i * sigma_y) =")
print(Exp(1j*sigma_y))
print("exp(i * sigma_z) = ")
print(Exp(1j*sigma_z))

print("cos(1) + i * sin(1) * sigma_x = ")
print(np.cos(1) * np.identity(2) + 1j * np.sin(1) * sigma_x)
print("cos(1) + i * sin(1) * sigma_y = ")
print(np.cos(1) * np.identity(2) + 1j * np.sin(1) * sigma_y)
print("cos(1) + i * sin(1) * sigma_z = ")
print(np.cos(1) * np.identity(2) + 1j * np.sin(1) * sigma_z)