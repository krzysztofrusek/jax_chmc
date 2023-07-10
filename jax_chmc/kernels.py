from typing import Callable, NamedTuple, Any

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular, cholesky
from jaxtyping import PyTree, Array, Float, PRNGKeyArray

from jax_chmc.newton import newton_solver, newton_solve


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


class RattleVars(NamedTuple):
    p_1_2: Array  # Midpoint momentum
    q_1: Array  # Midpoint position
    p_1: Array  # final momentum
    lam: Array  # Midpoint Lagrange multiplier (state)
    mu: Array  # final Lagrange multiplier (momentum)


class PQ(NamedTuple):
    p: Array
    q: Array
    # dc: Array


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

    def make_hamiltonian(logdensity_fn: Callable,sim=False):
        def hamiltonian(p: Array, q: Array):
            dc = j_con_fun(q)
            if sim:
                return 0.5 * p.T @ mass.inverse @ p - logdensity_fn(q)
            else:
                return 0.5 * p.T @ mass.inverse @ p + mass.compute_log_norm_const(dc) - logdensity_fn(q)

        return hamiltonian

    def rattle_integrator(s: PQ) -> PQ:
        """RATTLE integrator"""
        dc = j_con_fun(s.q)
        p0, q0 = s

        dH_dq = jax.grad(make_hamiltonian(sim_logdensity_fn, sim=True), argnums=1)
        dH_dp = jax.grad(make_hamiltonian(sim_logdensity_fn, sim=True), argnums=0)

        # TODO split into kinetic and potential energy
        def eq(x: RattleVars):
            C_q_1 = j_con_fun(x.q_1)
            zero = (
                p0 - step_size * 0.5 * ((dc.T @ x.lam) + dH_dq(x.p_1_2, q0)) - x.p_1_2,
                q0 - step_size * 0.5 * (dH_dp(x.p_1_2, q0) + dH_dp(x.p_1_2, x.q_1)) - x.q_1,
                con_fn(x.q_1),
                x.p_1_2 - step_size * 0.5 * (dH_dq(x.p_1_2, x.q_1) + (C_q_1.T @ x.mu)) - x.p_1,
                C_q_1 @ dH_dp(x.p_1, x.q_1)
            )
            return zero

        init_vars = RattleVars(p_1_2=p0,
                               q_1=q0 + mass.inverse @ p0,
                               p_1=p0,
                               lam=jnp.ones(dc.shape[0]),
                               mu=jnp.ones(dc.shape[0])
                               )

        sol = newton_solver(eq, init_vars, 8)
        #sol = newton_solve(lambda x,_: eq(x), init_vars, 8)
        return PQ(p=sol.x.p_1, q=sol.x.q_1)

    def kernel(
            rng_key: PRNGKeyArray,
            state: CHMCState,
    ) -> tuple[CHMCState, CHMCInfo]:
        proposal_key, accept_key = jax.random.split(rng_key)

        p0 = generare_momentum(state, proposal_key)

        target_H = make_hamiltonian(logdensity_fn, sim=False)

        pq0 = PQ(p0, state.position)
        pqL, _ = jax.lax.scan(lambda x, _: (rattle_integrator(x), None), pq0, xs=None, length=num_integration_steps)

        H0 = target_H(*pq0)
        H = target_H(*pqL)
        accept_p = jnp.minimum(1., jnp.exp(-(H - H0)))
        accept = jax.random.bernoulli(accept_key, accept_p)

        info = CHMCInfo(acceptance_rate=accept_p, is_accepted=accept,
                        proposal=pqL, momentum=pqL.p, is_divergent=None, energy=H,
                        num_integration_steps=num_integration_steps)

        # TODO improve performance
        new_state = jax.lax.cond(info.is_accepted, lambda: init(pqL.q), lambda: state)

        return new_state, info

    return SamplingAlgorithm(init, kernel)
