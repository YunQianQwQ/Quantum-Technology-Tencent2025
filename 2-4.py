import numpy as np

def H(n):
    I = np.identity(2)
    X = np.array([[0,1],[1,0]])
    Z = np.array([[1,0],[0,-1]])

    res = np.zeros((2**n,2**n))

    for i in range(n):
        mat = np.identity(1)
        for j in range(n):
            if i == j :
                mat = np.kron(mat,Z)
            else :
                mat = np.kron(mat,I)
        res = res + mat

    for i in range(n-1):
        mat1 = np.identity(1)
        for j in range(n):
            if i == j :
                mat1 = np.kron(mat1,X)
            else :
                mat1 = np.kron(mat1,I)

        mat2 = np.identity(1)
        for j in range(n):
            if i + 1 == j :
                mat2 = np.kron(mat2,X)
            else :
                mat2 = np.kron(mat2,I)
        
        res = res + np.dot(mat1,mat2)

    return res

def main():
    n = int(input())
    mat = H(n)
    print(mat[0][0])

if __name__ == "__main__":
    main()