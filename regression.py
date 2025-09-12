from scipy import io
import numpy as np

data = io.loadmat("./dataset.mat")
X = data['X']
D = data['D']

W_star = np.linalg.inv(X.T @ X) @ (X.T @ D)


Y = X @ W_star
MSE = np.mean((Y - D)**2)

N = X.shape[0]

print("Optimal W* =", W_star.flatten())
print("MSE =", MSE)





