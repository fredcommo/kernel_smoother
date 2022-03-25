import numpy as np

def bandwidth(y):
    """
    Estimate the optimal bandwidth to use by the kernel

    Parameters:
    ----------
    y : np.ndarray
        the array to smooth

    Raises:
    ------
        Exception, in case the variance of y is Zero

    Returns:
    -------
        a float
    """
    n = len(y)
    s = np.std(y)
    if s == 0:
        raise Exception("Bandwidth can't be estimated from Zero variance!")
    q75, q25 = np.percentile(y, [75 ,25])
    iqr = q75 - q25
    return 1.1*np.min([s, iqr/1.34])*n**(-1/5)