import sys
sys.path.append("src/kernel_smoother")

import numpy as np
from estimate_bandwidth import bandwidth
from kernels import *

def kern_smooth(x, y, K="gaussian", h=None):
    """
    Perform a kernel-based smoothing

    parameters:
    ----------
    x : np.ndarray
        the x values
    y : np.ndarray
        the y values to be smoothed
    K : str
        the name of the kernel method to be applied,
            one of the following methods: "gaussian", "triangular", "Epanechnikov"
    h : int or float
        the bandwidth

    Raises:
    ------
        Exception, in case of unsupported kernel name

    Returns:
    -------
        a np.ndarray of y-smoothed values
    """

    def _smooth(w, y):
        return 1/sum(w) * w.dot(y)

    valid_kernels = ["gaussian", "triangular", "Epanechnikov"]
    if K not in valid_kernels:
        error_msg = "kernel must be one of the following: {}" \
            .format(", ".join(valid_kernels))
        raise Exception(error_msg)

    if h is None:
        h = bandwidth(y)
    W = [eval(K)(x0, x, h) for x0 in x]
    return np.array([_smooth(w, y) for w in W])



if __name__ == "__main__": # pragma: no cover
    import matplotlib.pyplot as plt

    x = np.linspace(0, 10, num=200)
    noise = np.random.normal(loc=0, scale=0.25, size=len(x))
    y = np.sin(x) + noise


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(27, 8))

    ############################################
    # Compare banwidth effect on gaussian kernel
    bwidths = [0.1, bandwidth(y), 0.5, 1, 2]
    ax1.scatter(x, y, s=10, c='grey')
    for h in bwidths:
        ysmooth = kern_smooth(x, y, h=h)
        ax1.plot(x, ysmooth, linewidth=3)
    ax1.set_title("Bandwidth effect", fontsize=22)
    ax1.legend([f"h = {h}" for h in bwidths], title="Gaussian")
    ax1.set(xlabel='x', ylabel='y')


    ############################################
    # Compare kernels
    kernels = {
        "gaussian": bandwidth(y),
        "triangular": 1,
        "Epanechnikov": 1
        }

    ax2.scatter(x, y, s=10, c='grey')
    for kernel in kernels:
        ysmooth = kern_smooth(x, y, K=kernel, h=kernels[kernel])
        ax2.plot(x, ysmooth, linewidth=3)
    ax2.set_title("Kernel effect", fontsize=22)
    ax2.legend(kernels, title="Kernels")
    ax2.set(xlabel='x', ylabel='y')
    
    ############################################
    # Compare automated bandwidth on different variances

    x = np.linspace(0, 10, num=200)
    S = [0.1, 0.3, 0.5]
    for i, s in enumerate(S):
        noise = np.random.normal(loc=0, scale=s, size=len(x))
        y = np.sin(x) + i*3 + noise
        h = bandwidth(y)
        ysmooth = kern_smooth(x, y, K="gaussian", h=h)
        ax3.scatter(x, y, s=10, c='grey')
        ax3.plot(x, ysmooth, linewidth=3)
        ax3.annotate("Found bdw: {:.3f}".format(h), (0, i*3 - 1), fontsize=14)
    ax3.set_title("y-variance effect", fontsize=22)
    ax3.legend(S, title="Variances")
    ax3.set(xlabel='x', ylabel='y')

    plt.savefig('./demo_plot.png')
