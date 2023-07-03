import jax.numpy as jnp
import numpy as np


def conditional_mvn(loc, cov, a):
    """
    Conditional; distribution https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
    :param loc: location
    :param cov: covaraince
    :param a: vector we condition on. every `np.nan` is considered unknown
    :return: loc and covariance
    """
    unknown_idx, = np.where(jnp.isnan(a))
    known_idx = np.setdiff1d(np.arange(len(a)), unknown_idx)
    mu1 = loc[unknown_idx]
    mu2 = loc[known_idx]
    sigma11 = cov[unknown_idx, :][:, unknown_idx]
    sigma22 = cov[known_idx, :][:, known_idx]
    sigma21 = cov[known_idx, :][:, unknown_idx]
    sigma12 = cov[unknown_idx, :][:, known_idx]

    mubar = mu1 - sigma12 @ np.linalg.solve(sigma22, a[known_idx] - mu2)
    sigmabar = sigma11 - sigma12 @ np.linalg.solve(sigma22, sigma21)
    return mubar, sigmabar


def constrained_mvn(loc, cov, A):
    """Compute loc and covariance os a mvn distribution constrained to  :math:`A@x=0`
    First we make a joint distribution od :math:`x` and the constraints.
    Next the constraints are marginalized by means of  ``conditional_mvn``.

    :param loc: lcoation
    :param cov: covariance matrix
    :param A:constrain vector
    :return:loc and covariance
    """
    A = np.atleast_2d(A)
    Aall = np.concatenate([np.eye(len(loc)), A], axis=0)
    loc_hat = Aall @ loc
    cov_hat = Aall @ cov @ Aall.T
    a = np.ones_like(loc_hat)[:, 0] + jnp.nan
    a[-1] = 0
    mubar, sigmabar = conditional_mvn(loc_hat, cov_hat, a)
    return mubar, sigmabar
