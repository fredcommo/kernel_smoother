import sys
sys.path.append("src")
# sys.path.insert(0, '../src')

import pytest
import numpy as np
import pandas as pd
from src.kernel_smoother.kernels import gaussian, triangular, Epanechnikov


class Test_Gaussian:
    
    @pytest.mark.parametrize(
        "x0, x, h",
        [
            (2, np.array([0, 1, 2, 3, 4]), 0.1),
            (2, np.array([0, 1, 2, 3, 4]), 1),
            (2, np.array([0, 1, 2, 3, 4]), 10)
        ]
    )
    def test_gaussian_no_failure(self, x0, x, h):
        output = gaussian(x0, x, h=h)
        msg = "Expected 'numpy.ndarray, but got {}'"
        assert isinstance(output, np.ndarray), msg.format(type(output))

    @pytest.mark.parametrize(
        "x0, x, h",
        [
            (2, [0, 1, 2, 3, 4], 0.1),
            (2, (0, 1, 2, 3, 4), 0.1),
            (2, pd.Series([0, 1, 2, 3, 4]), 0.1),
            (2, np.array([1, 2, 3, 4]), 0)
        ],
        ids=[
            "x is a list",
            "x is a tuple",
            "x is a pd.Series",
            "h is not >0"
        ]
    )
    def test_gaussian_failures(self, x0, x, h):
        with pytest.raises(Exception):
            gaussian(x0, x, h=h)

class Test_Triangular:
    
    @pytest.mark.parametrize(
        "x0, x, h",
        [
            (2, np.array([0, 1, 2, 3, 4]), 0.1),
            (2, np.array([0, 1, 2, 3, 4]), 1),
            (2, np.array([0, 1, 2, 3, 4]), 10)
        ]
    )
    def test_triangular_no_failure(self, x0, x, h):
        output = triangular(x0, x, h=h)
        msg = "Expected 'numpy.ndarray, but got {}'"
        assert isinstance(output, np.ndarray), msg.format(type(output))

    @pytest.mark.parametrize(
        "x0, x, h",
        [
            (2, [0, 1, 2, 3, 4], 0.1),
            (2, (0, 1, 2, 3, 4), 0.1),
            (2, pd.Series([0, 1, 2, 3, 4]), 0.1),
            (2, np.array([1, 2, 3, 4]), 0)
        ],
        ids=[
            "x is a list",
            "x is a tuple",
            "x is a pd.Series",
            "h is not >0"
        ]
    )
    def test_triangluar_failures(self, x0, x, h):
        with pytest.raises(Exception):
            triangular(x0, x, h=h)


class Test_Triangular:
    
    @pytest.mark.parametrize(
        "x0, x, h",
        [
            (2, np.array([0, 1, 2, 3, 4]), 0.1),
            (2, np.array([0, 1, 2, 3, 4]), 1),
            (2, np.array([0, 1, 2, 3, 4]), 10)
        ]
    )
    def test_triangular_no_failure(self, x0, x, h):
        output = triangular(x0, x, h=h)
        msg = "Expected 'numpy.ndarray, but got {}'"
        assert isinstance(output, np.ndarray), msg.format(type(output))

    @pytest.mark.parametrize(
        "x0, x, h",
        [
            (2, [0, 1, 2, 3, 4], 0.1),
            (2, (0, 1, 2, 3, 4), 0.1),
            (2, pd.Series([0, 1, 2, 3, 4]), 0.1),
            (2, np.array([1, 2, 3, 4]), 0)
        ],
        ids=[
            "x is a list",
            "x is a tuple",
            "x is a pd.Series",
            "h is not >0"
        ]
    )
    def test_triangluar_failures(self, x0, x, h):
        with pytest.raises(Exception):
            triangular(x0, x, h=h)


class Test_Epanechnikov:
    
    @pytest.mark.parametrize(
        "x0, x, h",
        [
            (2, np.array([0, 1, 2, 3, 4]), 0.1),
            (2, np.array([0, 1, 2, 3, 4]), 1),
            (2, np.array([0, 1, 2, 3, 4]), 10)
        ]
    )
    def test_Epanechnikov_no_failure(self, x0, x, h):
        output = Epanechnikov(x0, x, h=h)
        msg = "Expected 'numpy.ndarray, but got {}'"
        assert isinstance(output, np.ndarray), msg.format(type(output))

    @pytest.mark.parametrize(
        "x0, x, h",
        [
            (2, [0, 1, 2, 3, 4], 0.1),
            (2, (0, 1, 2, 3, 4), 0.1),
            (2, pd.Series([0, 1, 2, 3, 4]), 0.1),
            (2, np.array([1, 2, 3, 4]), 0)
        ],
        ids=[
            "x is a list",
            "x is a tuple",
            "x is a pd.Series",
            "h is not >0"
        ]
    )
    def test_Epanechnikov_failures(self, x0, x, h):
        with pytest.raises(Exception):
            Epanechnikov(x0, x, h=h)
        