from typing import Callable, NamedTuple, Any

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular, cholesky
from jaxtyping import PyTree, Array, Float, PRNGKeyArray

from jax_chmc.rattle import Rattle
from diffrax import NewtonNonlinearSolver
import jax.tree_util as jtu
from diffrax import ODETerm, SaveAt
from equinox.internal import ω
import diffrax


class CHMCState(NamedTuple):
    """State of the CHMC algorithm.

    The CHMC algorithm takes one position of the chain and returns another
    position. In order to make computations more efficient, we also store
    the current logdensity as well as the current gradient of the logdensity and the
    constraint jacobian.

    """
    position: PyTree
    # hamiltonian:float
    # logdensity: float
    # logdensity_grad: ArrayTree
    constrain_jac: Array


class CHMCInfo(NamedTuple):
    momentum: PyTree
    acceptance_rate: float
    is_accepted: bool
    is_divergent: bool
    energy: float
    proposal: Any
    num_integration_steps: int


class Mass:
    cholesky: Array
    inverse: Array

    def __init__(self, M: Array):
        self.cholesky = cholesky(M)
        self.inverse = solve_triangular(self.cholesky.T, solve_triangular(self.cholesky, jnp.eye(*M.shape), lower=True),
                                        lower=False)

    def compute_log_norm_const(self, dc: Array) -> Float:
        # https://math.stackexchange.com/questions/3155163/computing-the-pdf-of-a-low-rank-multivariate-normal

        D = dc @ self.inverse
        cholMhat = self.cholesky - D.T @ jnp.linalg.solve(D @ D.T, D @ self.cholesky)
        d = jnp.linalg.svd(cholMhat, compute_uv=False, hermitian=True)
        top_d, _ = jax.lax.top_k(d, dc.shape[1] - dc.shape[0])
        return jnp.sum(jnp.log(top_d))



class SamplingAlgorithm(NamedTuple):
    init: Callable
    step: Callable


def fun_chmc(
        logdensity_fn: Callable,  # H
        sim_logdensity_fn: Callable,  # hat H
        con_fn: Callable,
        step_size,  # h
        inverse_mass_matrix,  # M
        num_integration_steps,  # L
) -> SamplingAlgorithm:
    mass = Mass(inverse_mass_matrix)
    j_con_fun = jax.jacobian(con_fn)

    def generare_momentum(state: CHMCState, proposal_key):
        z = jax.random.normal(proposal_key, shape=state.position.shape)
        p0 = mass.cholesky @ z
        # projection
        dc = state.constrain_jac
        D = dc @ mass.inverse
        p0 = p0 - D.T @ jnp.linalg.solve(D @ D.T, D @ p0)

        return p0

    def init(position: PyTree) -> CHMCState:
        f, df = jax.value_and_grad(logdensity_fn)(position)
        jac = j_con_fun(position)

        return CHMCState(position=position,
                         # hamiltonian=0.,# hamiltonian be pendu
                         # logdensity_grad=df,
                         constrain_jac=jac
                         )

    def make_hamiltonian(logdensity_fn: Callable, sim=False):
        def hamiltonian(p: Array, q: Array):
            dc = j_con_fun(q)
            if sim:
                return 0.5 * p.T @ mass.inverse @ p - logdensity_fn(q)
            else:
                return 0.5 * p.T @ mass.inverse @ p + mass.compute_log_norm_const(dc) - logdensity_fn(q)

        return hamiltonian


    def kernel(
            rng_key: PRNGKeyArray,
            state: CHMCState,
    ) -> tuple[CHMCState, CHMCInfo]:
        proposal_key, accept_key = jax.random.split(rng_key)

        p0 = generare_momentum(state, proposal_key)

        target_H = make_hamiltonian(logdensity_fn, sim=False)

        pq0 = (p0, state.position)

        rat = Rattle(nonlinear_solver=NewtonNonlinearSolver(rtol=1e-4, atol=1e-6), constrain=con_fn)
        H = make_hamiltonian(sim_logdensity_fn, sim=True)
        terms = (ODETerm(lambda t, q, args: (-jax.grad(H, argnums=1)(jtu.tree_map(jnp.zeros_like, q), q) ** ω).ω),
                 ODETerm(lambda t, p, args: jax.grad(H, argnums=0)(p, jtu.tree_map(jnp.zeros_like, p))))
        saveat = SaveAt(t1=True)
        dt = 0.1
        t1 = num_integration_steps * dt
        solution = diffrax.diffeqsolve(terms, rat, 0.0, t1, dt0=dt, y0=pq0, saveat=saveat)
        pqL = (solution.ys[0][0], solution.ys[1][0])

        H0 = target_H(*pq0)
        H = target_H(*pqL)
        accept_p = jnp.minimum(1., jnp.exp(-(H - H0)))
        accept = jax.random.bernoulli(accept_key, accept_p)

        info = CHMCInfo(acceptance_rate=accept_p, is_accepted=accept,
                        proposal=pqL, momentum=pqL[0], is_divergent=None, energy=H,
                        num_integration_steps=num_integration_steps)

        # TODO improve performance
        new_state = jax.lax.cond(info.is_accepted, lambda: init(pqL[1]), lambda: state)

        return new_state, info

    return SamplingAlgorithm(init, kernel)
