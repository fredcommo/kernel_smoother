import sys
sys.path.append("src")

import pytest
import numpy as np
from src.kernel_smoother.kernels import gaussian, triangular, Epanechnikov
from src.kernel_smoother.smoother import kern_smooth

@pytest.mark.parametrize(
    "h, scale, kernel",
    [
        (0.01, 0.01, "gaussian"),
        (0.1, 0.02, "gaussian"),
        (1, 0.1, "gaussian"),
        (0.01, 0.01, "triangular"),
        (0.1, 0.02, "triangular"),
        (1, 0.1, "triangular"),
        (0.01, 0.01, "Epanechnikov"),
        (0.1, 0.02, "Epanechnikov"),
        (1, 0.1, "Epanechnikov")
        ]
)
def test_kern_smooth_no_failure(h, scale, kernel):
    x = np.linspace(0, 10, num=200)
    y = np.sin(x) + np.random.normal(loc=0, scale=scale, size=len(x))
    ysmooth = kern_smooth(x, y, K=kernel, h=h)
    assert isinstance(ysmooth, np.ndarray), f"Ouput is {type(ysmooth)}, while expected a 'np.ndarray'"
    assert np.allclose(y, ysmooth, atol=h)

def test_kern_smooth_bad_kernel():
    x = np.linspace(0, 10, num=200)
    y = np.sin(x) + np.random.normal(loc=0, scale=1, size=len(x))
    with pytest.raises(Exception):
        kern_smooth(x, y, K="foo", h=1)
    