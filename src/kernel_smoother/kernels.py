import numpy as np

def gaussian(x0, x, h=0.25):
    """
    Gaussian kernel
    
    Parameters:
    ----------
    x0 : int or float
        the value to be evaluated
    x : np.ndarray
        the full vector of values
    h : int or float
        the bandwidth

    Raises:
    ------
        Exception, in case h is not strictly > 0
        Exception, in case x is not a np.ndarray

    Returns:
    -------
        A np.ndarray of weights
    """

    if h <= 0:
        raise Exception("Bandwidth ('h') must be strictly positive")
    if not isinstance(x, np.ndarray):
        raise Exception(f"'x' must be a numpy.ndarray, got {type(x)} instead")
    u = (x - x0) / h
    return 1/(2*np.pi) * np.exp(-u**2 / 2)


def triangular(x0, x, h=1):
    """
    Triangular kernel
    
    Parameters:
    ----------
    x0 : int or float
        the value to be evaluated
    x : np.ndarray
        the full vector of values
    h : int or float
        the bandwidth

    Raises:
    ------
        Exception, in case h is not strictly > 0
        Exception, in case x is not a np.ndarray

    Returns:
    -------
        A np.ndarray of weights
    """

    def _f(u):
        return 1 - u if u <= 1 else 0
    
    if h <= 0:
        raise Exception("Bandwidth ('h') must be strictly positive")
    if not isinstance(x, np.ndarray):
        raise Exception(f"'x' must be a numpy.ndarray, got {type(x)} instead")
    U = np.abs(x - x0) / h
    return np.array([_f(u) for u in U])


def Epanechnikov(x0, x, h=1):
    """
    Epanechnikov kernel
    
    Parameters:
    ----------
    x0 : int or float
        the value to be evaluated
    x : np.ndarray
        the full vector of values
    h : int or float
        the bandwidth

    Raises:
    ------
        Exception, in case h is not strictly > 0
        Exception, in case x is not a np.ndarray

    Returns:
    -------
        A np.ndarray of weights    
    """

    def _f(u):
        return 3/4*(1 - u**2) if u <= 1 else 0
    
    if h <= 0:
        raise Exception("Bandwidth ('h') must be strictly positive")
    if not isinstance(x, np.ndarray):
        raise Exception(f"'x' must be a numpy.ndarray, got {type(x)} instead")
    U = np.abs(x - x0) / h
    return np.array([_f(u) for u in U])
