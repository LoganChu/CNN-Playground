from scipy import io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


data = io.loadmat("./dataset.mat")
X = data['X']
D = data['D']


W_star = np.linalg.inv(X.T @ X) @ (X.T @ D)


Y = X @ W_star

W = np.array([-0.2,0.0,0.45])
W = W.reshape(-1,1)

epochs = 20

learning_rate = 0.005
N = X.shape[0]

MSE_list = []

for i in range(epochs):

    for k in range(N):
        x_k = X[k, :].reshape(-1,1)
        d_k = D[k,0]

        predicted_val = W.T@x_k
        loss = d_k-predicted_val
        W = W + learning_rate*loss*x_k
    Y = X @ W
    MSE = np.mean((Y - D) ** 2)
    MSE_list.append(MSE)


# Final weights and MSE list
print("Final W after 20 epochs:", W.flatten())
print("Final MSE:", MSE_list[-1])

# Plot MSE (log scale)
plt.plot(range(1, epochs+1), MSE_list)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE (log scale)")
plt.title("LMS Training Curve")
plt.show()


fig = plt.figure()
ax = plt.axes(projection='3d')

xdata = X[:,1]
ydata = X[:,2]
zdata = D[:,0]


LMS_predicted = X@W

ax.scatter3D(xdata, ydata, zdata, c="blue")

ax.scatter3D(xdata, ydata, Y.flatten(), c="red")

ax.scatter3D(xdata, ydata, LMS_predicted.flatten(), c="green")

plt.show()