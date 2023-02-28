import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def normalizedError(X, X_pred): return np.linalg.norm(X-X_pred, 2)/np.linalg.norm(X, 2)

def generateA(M, N):
    A = np.random.normal(size=(M, N))
    A /= np.linalg.norm(A, axis=0)
    return A

def generateSparseS(N, cardinality):
    S = np.random.choice(range(N), cardinality, replace=False)
    x = np.random.uniform(0, 17, size=(cardinality, 1))
    x[np.where(x<=9)] -= 10
    x[np.where(x>9)] -= 8
    X = np.zeros(N)
    X[S[:, np.newaxis]] = x
    return X, S

def generateValues(M, N, cardinality, mean, sigma):
    A = np.random.normal(size=(M, N))
    A /= np.linalg.norm(A, axis=0)
   
    S = np.random.choice(range(N), cardinality, replace=False)
    x = np.random.uniform(0, 17, size=(cardinality, 1))
    x[np.where(x<=9)] -= 10
    x[np.where(x>9)] -= 8
    X = np.zeros(N)
    X[S[:, np.newaxis]] = x
    
    n = np.random.normal(mean, sigma, M)
    return A, X, n, S

@njit
def pseudoinverse(mat):
    return np.dot(np.linalg.inv(np.dot(mat.T, mat)), mat.T)

'''OMP'''
@njit
def omp(A, y, k):
    r, M, S, x = y.copy(), A.shape[1], set({}), []

    ''' known sparsity with noise'''
    '''
    for i in range(k):
        if np.linalg.norm(r)<0.0001:break
    '''

    '''unknown sparity with noise'''
    '''
    while(np.linalg.norm(r)>k):
    '''

    '''noiseless case'''
    while(np.linalg.norm(r)>0.001):

        lambda_k = np.argmax(np.abs(np.dot(r, A)))
        S.add(lambda_k)
        A_ = A[:, list(S)]
        x = np.dot(pseudoinverse(A_), y)
        # x = np.dot(np.linalg.pinv(A_), y) ## sparsity known noise
        r = y - np.dot(A_, x)
    x_ret = np.zeros(M)
    x_ret[list(S)] = x

    return x_ret, S

'''Plotting'''
S = 20
N_range = [20, 50, 100]
M = [20, 30, 30]
repeats = 2000
count_no_noise = []

for j, n in enumerate(N_range):
    count = np.zeros([2, M[j], S])
    for m in range(1, M[j] + 1):
        for s in range(1, S+1):
            for _ in range(repeats):
                A, X, noise, omega = generateValues(m, n, s, 0, 1)
                y = np.dot(A, X)
                X_pred, omega_pred = omp(A, np.dot(A, X), 0) # change the third value depending upon the noisy case
                count[0, m-1, s - 1] += normalizedError(X, X_pred) <= 0.001
                count[1, m-1, s - 1] += sorted(omega_pred) == sorted(omega)
    count /= repeats
    count_no_noise.append([count])

    fig, ax = plt.subplots(1, 2)
    for i in range(2):
        ax[i].imshow(count[i], cmap='gray')
        ax[i].set_xlabel('s')
        ax[i].set_ylabel('M')
    ax[0].set_title('Normalized Error')
    ax[1].set_title('Exact Matching')
    plt.show()

'''Decoding Image'''
X_pred_images = []
for i in range(1, 4):
    X_pred_images.append(omp(mat['A'+str(i)], mat['y'+str(i)][:, 0], 1))
    plt.imshow(X_pred_images[-1].reshape(160, 90).T, cmap='gray')
    plt.show()