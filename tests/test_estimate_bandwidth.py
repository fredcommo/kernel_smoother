import sys
# sys.path.append("src")
sys.path.insert(0, '../src')

import pytest
from kernel_smoother.estimate_bandwidth import bandwidth

@pytest.mark.parametrize(
    "y, expected",
    [
        ([1, 2, 3, 4, 5, 6], 1),
        ([10, 20, 30, 40, 50, 60], 10),
        ([100, 200, 300, 400, 500, 600], 100)
        ]
)
def test_estimate_bandwidth(y, expected):
    output = bandwidth(y)
    
    msg = "Expected float, but got {}"
    assert isinstance(output, float), msg.format(type(output))

    msg = "Expected {}, but got {:.2f}"
    assert output > expected, msg.format(expected, output)


def test_estimate_zero_variance():
    with pytest.raises(Exception):
        bandwidth([1]*10)
