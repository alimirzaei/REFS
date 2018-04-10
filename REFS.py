
import numpy as np
from scipy.spatial.distance import cdist
from keras.datasets import mnist
from tqdm import tqdm


# load training data
#X = np.random.randn(10,3)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X = x_test.reshape(10000, -1)
d = X.shape[1]
n = X.shape[0]

# parameters
alpha = .5   # other features coeficient
betha = .5   # graph coeficient 
p = 2        # number of neighbors for graph
k = 3        # number of selected features


# calclate adjusment matrix
distance_marix = cdist(X.T, X.T)
W = np.zeros_like(distance_marix)

print('Start to Calculate Adjusment Matrix')
max_distance = np.zeros(d)
for i in range(d):
    max_distance[i] = sorted(distance_marix[i,:])[p]
for i in range(d):
    for j in range(d):
        if(distance_marix[i,j] <= max_distance[i] or distance_marix[i,j] <= max_distance[j]):
            W[i,j] = 1

# calculate Laplaian matrix
D = np.diag(np.sum(W, axis=1))
L = D - W

# initialization
S = np.array([])
i = 0
j = 1
gamma = 1 - alpha

Q_inv = list(np.zeros(d+1))
Q0 = alpha * np.eye(d) + betha * L
Q_inv[0] = np.linalg.inv(Q0)

print('Start Feature Selection')
bar = tqdm(total=d*k)
while(i < k):
    v = 1000*np.ones(d)
    while (j < d):
        if(j not in S):
            Q_inv[j] = Q_inv[i] - (gamma * Q_inv[i].dot(np.eye(d)[:,j:j+1]).dot(np.eye(d)[j:j+1,:]).dot(Q_inv[i])) / (1 + gamma*np.eye(d)[j:j+1, :].dot(Q_inv[i]).dot(np.eye(d)[:, j:j+1]))
            v[j] = np.linalg.norm(betha*X.dot(L).dot(Q_inv[j]))
        bar.update((i+1)*j)
        j = j + 1
    j = v.argmin()
    S = np.append(S, j)
    i = i + 1
    Q_inv[i] = Q_inv[j]








