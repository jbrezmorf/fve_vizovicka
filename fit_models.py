import pickle
from typing import List, Tuple

import attrs
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import interpolate

def estimate_time_shift(ts1, ts2):
    """
    Estimate the time shift (dt) between two time series using derivatives.
    dt > 0 means ts2 is shifted to the right relative to ts1.
    Parameters:
        ts1 (pd.Series): First time series.
        ts2 (pd.Series): Second time series.

    Returns:
        float: Estimated time shift (dt).
    """
    # Compute first differences (derivatives)
    d_f1 = np.diff(ts1.values)
    d_f2 = np.diff(ts2.values)

    # Compute mean derivative
    d_mean = (d_f1 + d_f2) / 2

    # Mask where abs(d_mean) is greater than the lower quartile
    mask = np.abs(d_mean) > 0.01

    # Select valid points for f1, f2, and d_mean (accounting for the difference size due to np.diff)
    f1_values = ts1.values[1:]  # First differences reduce the array size by 1
    f2_values = ts2.values[1:]

    f1_f2_diff = f1_values[mask] - f2_values[mask]
    d_mean_masked = d_mean[mask]

    # Estimate the time shift
    dt = np.mean(f1_f2_diff / d_mean_masked)

    return dt


def apply_fractional_shift(ts, shift):
    # Shift the index by the fractional amount
    shifted_index = ts.index + pd.to_timedelta(shift, unit="H")

    # Interpolate back to the original index
    ts_shifted = ts.reindex(ts.index.union(shifted_index)).interpolate(method='cubicspline').reindex(ts.index)
    return ts_shifted


def plot_2d_data(x, y, z):
    print(f"X range: {np.min(x)}, {np.max(x)}")
    print(f"Y range: {np.min(y)}, {np.max(y)}")
    print(f"Z range: {np.min(z)}, {np.max(z)}")
    # Plot function and sample points
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    #c = ax.contourf(x, y, z, 15, cmap=plt.cm.RdBu);
    scatter = ax.scatter(x, y, marker='.', c=z, cmap=plt.cm.RdBu)
    #ax.set_ylim(-1, 1)
    #ax.set_xlim(-1, 1)
    ax.set_xlabel(r"$x$", fontsize=20)
    ax.set_ylabel(r"$y$", fontsize=20)
    cb = fig.colorbar(scatter, ax=ax)
    cb.set_label(r"$z$", fontsize=20)
    sc2 =ax2.scatter(x, z, c=y, marker='.', label='x', cmap=plt.cm.RdBu)
    cb2 = fig.colorbar(sc2, ax=ax2)
    plt.show()


def approx_2d(workdir, in1, in2, out=None):
    """
    If 'out' is given:
        - Construct approximation
        - write down approximationg function
        - apply to inputs
    else:
        - read approximation
        - apply to inputs
    :param workdir:
    :param in1:
    :param in2:
    :param out:
    :return:
    """
    points = np.stack([in1, in2], axis=1)
    func_file = workdir / "dc_approx"
    if out is None:
        with open(func_file, 'rb') as f:
            func = pickle.load(f)
    else:
        values = out
        func = interpolate.CloughTocher2DInterpolator(points, values, tol=1e-2, rescale=True)
        with open(func_file, 'wb') as f:
            pickle.dump(func, f)
    return func(points), func


@attrs.define
class approx_fun:
    basis: List[Tuple[int, int]]
    coeffs: np.ndarray

    def __call__(self, X, y):
        # Define the callable object
        X = np.atleast_2d(X)
        X, y = np.asarray(X), np.asarray(y)
        if X.shape[1] != y.shape[0]:
            raise ValueError("x and y must have the same shape.")
        # Evaluate the polynomial
        return sum(c * np.sum(X ** px, axis=0) * (y ** py) for c, (px, py) in zip(self.coeffs, self.basis))


def least_squares_fit(X, Y, Z, basis):
    """
    Perform a least squares fit for Z as a function of X and Y using a specified polynomial basis.

    Parameters:
        X (array-like): Input X values.
        Y (array-like): Input Y values.
        Z (array-like): Output Z values (target).
        basis (list of tuple): List of (px, py) pairs where px and py are powers of X and Y.

    Returns:
        callable: A function fun(x, y) for evaluating the fitted surface.
    """
    # Validate input
    X = np.atleast_2d(X)   # (2, N)
    Y, Z = np.asarray(Y), np.asarray(Z) #(N,)
    if X.shape[1] != Y.shape[0] or X.shape[1] != Z.shape[0]:
        raise ValueError("X, Y, and Z must have the same shape.")

    # Build the design matrix
    A = np.column_stack([np.sum(X**px,axis=0) * Y**py for px, py in basis])

    # Solve the least squares problem
    coeffs, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    print("Coeffs: ", *zip(basis, coeffs))
    return approx_fun(basis, coeffs)


def lin_2d_approx(in1, in2, out=None):
    base = [(0,0), (1,0), (0,1), (1,1), (2, 0), (0, 2)]
    func = least_squares_fit(in1, in2, out, base)
    return func
    # X = np.stack([np.ones(len(in1)), in1, in2, in1**2, in2**2, in1*in2], axis=1)
    # func_file = workdir / "dc_approx"
    # if out is None:
    #     with open(func_file, 'rb') as f:
    #         beta = pickle.load(f)
    # else:
    #     beta = np.linalg.lstsq(X.T, out)[0]
    # func = lambda dc, temp : beta[0] + beta[1] * dc + beta[2] * temp
    # return func(in1, in2), func
