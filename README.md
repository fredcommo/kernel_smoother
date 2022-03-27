# To perform kernel smoothing

![Test-Badge](https://github.com/fredcommo/kernel_smoother/actions/workflows/CI.yml/badge.svg)

![Test-Report](https://github.com/fredcommo/kernel_smoother/blob/c85b5bea94cccf19c1f8ad2a647820f60ee5dfd7/tests/logs/tests-logs-ubuntu-latest-py3.8.txt)

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

### Other possible outputs:

![illustrations](demo_plot.png "Some examples of kernel smoothing")