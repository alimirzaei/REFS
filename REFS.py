# Author : Ali Mirzaei
# Date : 20 / 02 / 1397


import numpy as np
from scipy.spatial.distance import cdist
from keras.datasets import mnist
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.io import loadmat

    
# load training data
#X = np.random.randn(10,3)
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = np.abs(loadmat('/home/ali/My_noisy_H_22.mat')['My_noisy_H'])
X = x_test.reshape(40000, -1)[:10000,:]
d = X.shape[1]
n = X.shape[0]

# parameters
alpha = 0.9    # other features coeficient
betha = 0.1    # graph coeficient 
p = 5          # number of neighbors for graph
k = 5          # number of selected features


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
j = 0
gamma = 1 - alpha

Q_inv = list(np.zeros(d+1))
Q0 = alpha * np.eye(d) + betha * L
Q_inv[0] = np.linalg.inv(Q0)

print('Start Feature Selection')
bar = tqdm(total=k)
while(i < k):
    v = np.inf*np.ones(d)
    Q_inv_j = list(np.zeros(d))
    for j in range(d):
        if(j not in S):
            Q_inv_j[j] = Q_inv[i] - (gamma * Q_inv[i].dot(np.eye(d)[:,j:j+1]).dot(np.eye(d)[j:j+1,:]).dot(Q_inv[i])) / (1 + gamma*np.eye(d)[j:j+1, :].dot(Q_inv[i]).dot(np.eye(d)[:, j:j+1]))
            v[j] = np.linalg.norm(betha*X.dot(L).dot(Q_inv_j[j]))
    f = v.argmin()
    S = np.append(S, f)
    i = i + 1
    bar.update(i)
    Q_inv[i] = Q_inv_j[f]

print("Selected Features Are : ")
print(S)


def reconstruct(X_test):
    pi_Xs = np.zeros((len(X_test), d))
    pi_Xs[:, S.astype(int)] = X_test[:, S.astype(int)]
    reconstructed = (gamma* pi_Xs + alpha*X_test).dot(np.linalg.inv(gamma*np.linalg.inv(X_test.T.dot(X_test)).dot(X_test.T.dot(pi_Xs))+alpha*np.eye(len(L))+betha*L))
    return reconstructed


features = np.zeros(14*72)
features[S.astype(int)] = 1
fig = plt.figure()

#ax.imshow(features.reshape(28,-1))

X_recons = reconstruct(X)
for i in range(4):
    ax = fig.add_subplot(4, 2, 2*i+1)
    ax.imshow(X[i,:].reshape(72, -1))
    ax = fig.add_subplot(4, 2, 2*i+2)
    ax.imshow(X_recons[i,:].reshape(72, -1))

fig.show()

fig2 = plt.figure()
ax = fig2.add_subplot(1, 1, 1)
ax.imshow(features.reshape(72,-1))
fig2.show()

fig3 = plt.figure()
ax = fig3.gca(projection='3d')
X, Y = np.meshgrid(range(14), range(72))
Z1 = X_recons[0,:].reshape(72, -1)
Z2 = X[0,:].reshape(72, -1)
ax.plot_surface(X,Y,Z1)

fig4 = plt.figure()
ax = fig4.gca(projection='3d')
ax.plot_surface(X,Y,Z2)


