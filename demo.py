import numpy as np
from kernel_smoother.smoother import kern_smooth

np.random.seed(1)

x = np.linspace(0, 10, num=200)
noise = np.random.normal(loc=0, scale=0.25, size=len(x))
y = np.sin(x) + noise

kernels = ["gaussian", "triangular", "Epanechnikov"]
for k in kernels:
    y_smooth = kern_smooth(x, y, K=k)
    rmse = np.mean((y_smooth - y)**2)*(1/2)
    print("Kernel: {}, RMSE: {:.3f}, Valid: {}" \
        .format(k, rmse, rmse < 0.03))

# plt.scatter(x, y, s=10, c='grey')

# kernels = ["gaussian", "triangular", "Epanechnikov"]
# for k in kernels:
#     plt.plot(x, kern_smooth(x, y, K=k), linewidth=3)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(kernels, title="Kernels")
# plt.show()