from typing import Callable

import chex
import jax
from chex import ArrayTree, Scalar, Array
from jax import numpy as jnp
from jax._src.flatten_util import ravel_pytree


@chex.dataclass
class NewtonState:
    x: ArrayTree
    delta: ArrayTree
    n: Scalar
    aux:ArrayTree=None


def newton_solver(fun: Callable, x0: ArrayTree,
                  max_iter: int,
                  min_norm: Scalar = 1e-5,
                  has_aux=False) -> NewtonState:
    """
    Solve nonlinear equation :math:`fun(x)=0` by iterative Newton method starting from :math:`x0`
    :param max_iter: Maximum number of iteration
    :param min_norm: Minimal norm of the update
    :param fun: A function
    :param x0: Starting point
    :return:
    """
    x0_raveled, unravel_fn = ravel_pytree(x0)

    def _f(x):
        o = fun(unravel_fn(x))
        if has_aux:
            o, aux = o
            y, _ = ravel_pytree(o)
            return y, aux
        else:
            y, _ = ravel_pytree(o)
            return y

    def step_fun(x: NewtonState) -> NewtonState:
        jf = jax.jacobian(_f,has_aux=has_aux)

        J = jf(x.x)
        F = _f(x.x)
        if has_aux:
            J,_ = J
            F,aux = F
        else:
            aux=None

        delta = jnp.linalg.solve(J, -F)
        return NewtonState(x=x.x + delta,
                           delta=delta,
                           n=x.n + jnp.ones_like(x.n),
                           aux=aux)

    def cond(x: NewtonState):
        flat_delta, _ = ravel_pytree(x.delta)
        return jnp.logical_and(x.n < max_iter,
                               jnp.linalg.norm(flat_delta) > min_norm
                               )
    if has_aux:
        _,aux = _f(x0_raveled)
    else:
        aux =None

    sol = jax.lax.while_loop(cond, step_fun,
                             NewtonState(x=x0_raveled,
                                         delta=x0_raveled,
                                         n=jnp.zeros((), dtype=jnp.int32),
                                         aux=aux
                                         )
                             )
    return sol.replace(x=unravel_fn(sol.x), delta=unravel_fn(sol.delta))


def vector_newton_solver(fun: Callable, x0: Array, max_iter: int, min_norm: Scalar = 1e-5) -> NewtonState:
    """
    Solve nonlinear equation :math:`fun(x)=0` by iterative Newton method starting from :math:`x0`
    :param max_iter: Maximum number of iteration
    :param min_norm: Minimal norm of the update
    :param fun: A function
    :param x0: Starting point
    :return:
    """

    def step_fun(x: NewtonState) -> NewtonState:
        J = jax.jacobian(fun)(x.x)
        F = fun(x.x)
        delta = jnp.linalg.solve(J, -F)
        return NewtonState(x=x.x + delta, delta=delta, n=x.n + jnp.ones_like(x.n))

    def cond(x: NewtonState):
        return jnp.logical_and(x.n < max_iter,
                               jnp.linalg.norm(x.delta) > min_norm
                               )

    sol = jax.lax.while_loop(cond, step_fun, NewtonState(x=x0, delta=x0, n=jnp.zeros((), dtype=jnp.int32)))
    return sol
