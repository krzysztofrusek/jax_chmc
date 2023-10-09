import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
import jax

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


@jax.jit
def static_conditional_mvn(loc:Array, cov:Array, a:Array)->tuple[Array,Array]:
    """
    Calculates parameter of MVN conditioned on event that the last dimentions are equal to a
    :param loc:
    :param cov:
    :param a:
    """
    q = a.shape[0]
    mu1 = loc[:-q]
    mu2=loc[-q:]

    sigma11 = cov[:-q,:-q]
    sigma22 = cov[-q:, -q:]
    sigma21 = cov[-q:, :-q]
    sigma12 = cov[:-q, -q:]

    mubar = mu1 - sigma12 @ jnp.linalg.solve(sigma22, a - mu2)
    sigmabar = sigma11 - sigma12 @ jnp.linalg.solve(sigma22, sigma21)
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

@jax.jit
def static_constrained_mvn(loc:Array, cov:Array, A:Array,b:Array):
    """Compute loc and covariance os a mvn distribution constrained to  :math:`A@x=b`
    First we make a joint distribution od :math:`x` and the constraints.
    Next the constraints are marginalized by means of  ``conditional_mvn``.

    :param loc: lcoation
    :param cov: covariance matrix
    :param A:constrain vector
    :return:loc and covariance
    """
    A = jnp.atleast_2d(A)
    Aall = jnp.concatenate([jnp.eye(*cov.shape), A], axis=0)
    loc_hat = Aall @ loc
    cov_hat = Aall @ cov @ Aall.T
    return static_conditional_mvn(loc_hat, cov_hat, b)