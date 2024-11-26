#logistic_functions

import numpy as np

def logistic_curve(t, L, k, t0):
    """
    Logistic function.

    Parameters:
        t: Time or independent variable.
        L: Carrying capacity (the maximum value of the function).
        k: Growth rate (steepness of the curve).
        t0: Midpoint (time at which the function value is L/2).

    Returns:
        The output of the logistic function given the input parameters
    """
    return L / (1 + np.exp(-k * (t - t0)))

def logistic_derivative(t, L, k, t0):
    """
    First derivative of the logistic function.

    Parameters:
        t: Time or independent variable.
        L: Carrying capacity (the maximum value of the function).
        k: Growth rate (steepness of the curve).
        t0: Midpoint (time at which the function value is L/2).

    Returns:
        The output of the first derivative of the logistic function given the input parameters
    """
    exp_term = np.exp(-k * (t - t0))
    return (L * k * exp_term) / ((1 + exp_term) ** 2)

def logistic_second_derivative(t, L, k, t0):
    """
    Second derivative of the logistic function.

    Parameters:
        t: Time or independent variable.
        L: Carrying capacity (the maximum value of the function).
        k: Growth rate (steepness of the curve).
        t0: Midpoint (time at which the function value is L/2).

    Returns:
        The output of the second derivative of the logistic function given the input parameters
    """
    exp_term = np.exp(-k * (t - t0))
    return -(L * k**2 * exp_term * (1 - exp_term)) / ((1 + exp_term) ** 3)