# To perform kernel smoothing

![Test-Badge](https://github.com/fredcommo/kernel_smoother/actions/workflows/CI.yml/badge.svg)
[![python version](https://img.shields.io/badge/python-3.6%7C3.7%7C3.8%7C3.9-blue)](https://github.com/fredcommo/kernel_smoother.git)
[![Linux](https://svgshare.com/i/Zhy.svg)](https://svgshare.com/i/Zhy.svg)
[![Windows](https://svgshare.com/i/ZhY.svg)](https://svgshare.com/i/ZhY.svg)

## Install
Either clone the repo, or install from git as follows:

```python
python -m pip install git+https://github.com/fredcommo/kernel_smoother.git
```

## Quick example:

```python
import numpy as np
import matplotlib.pyplot as plt
from kernel_smoother.smoother import kern_smooth

x = np.linspace(0, 10, num=200)
noise = np.random.normal(loc=0, scale=0.25, size=len(x))
y = np.sin(x) + noise

plt.scatter(x, y, s=10, c='grey')

kernels = ["gaussian", "triangular", "Epanechnikov"]
for k in kernels:
    plt.plot(x, kern_smooth(x, y, K=k), linewidth=3)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(kernels, title="Kernels")
plt.show()
```

![illustrations](figures/Figure_1.png "Some examples of kernel smoothing")

### Other examples:

![illustrations](figures/Figure_2.png "Some examples of kernel smoothing")