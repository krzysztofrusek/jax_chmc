from typing import Tuple, Callable, NamedTuple

import diffrax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from diffrax import AbstractImplicitSolver
from diffrax import AbstractTerm
from diffrax.custom_types import Bool, DenseInfo, PyTree, Scalar, Array
from diffrax.local_interpolation import LocalLinearInterpolation
from diffrax.solution import RESULTS
from equinox.internal import ω

_ErrorEstimate = None
_SolverState = None


class RattleVars(NamedTuple):
    p_1_2: Array  # Midpoint momentum
    q_1: Array  # Midpoint position
    p_1: Array  # final momentum
    lam: Array  # Midpoint Lagrange multiplier (state)
    mu: Array  # final Lagrange multiplier (momentum)


class Rattle(AbstractImplicitSolver):
    """ Rattle method.

    Symplectic method. Does not support adaptive step sizing. Uses 1st order local
    linear interpolation for dense/ts output.
     dv = f(t,w(t)) dt
     dw = g(t,v(t))dt
     Hamiltonian
     H = V(p) + U(q)
     dp = -∂H/∂q dt
     dq = ∂H/∂p dt

    """

    term_structure = (AbstractTerm, AbstractTerm)
    interpolation_cls = LocalLinearInterpolation
    constrain: Callable = None  # Fix TypeError: non-default argument 'constrain' follows default argument

    def order(self, terms):
        return 2

    def init(self, terms: Tuple[AbstractTerm, AbstractTerm], t0: Scalar, t1: Scalar, y0: PyTree,
            args: PyTree, ) -> _SolverState:
        return None

    def step(self, terms: Tuple[AbstractTerm, AbstractTerm], t0: Scalar, t1: Scalar, y0: Tuple[PyTree, PyTree],
            args: PyTree, solver_state: _SolverState, made_jump: Bool, ) -> Tuple[
        Tuple[PyTree, PyTree], _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump

        term_1, term_2 = terms
        y0_1, y0_2 = y0  # p, q
        step_size = t1 - t0
        # term_1.contr(t0,t1)
        midpoint = (t1 + t0) / 2

        # term_1.contr(t0,midpoint) == 0.5*step_size
        # term_2.contr(t0,midpoint) == 0.5*step_size

        # p
        control1_half_1 = term_1.contr(t0, midpoint)
        control1_half_2 = term_1.contr(midpoint, t1)

        # q
        control2_half_1 = term_2.contr(t0, midpoint)
        control2_half_2 = term_2.contr(midpoint, t1)

        control2 = term_2.contr(t0, t1)

        p0 = y0_1
        q0 = y0_2

        p0, q0 = y0

        j_con_fun = jax.jacobian(self.constrain)
        dc = j_con_fun(q0)

        def dH_dq(p, q):
            return term_1.vf(0., q, args)

        def dH_dp(p, q):
            return -term_2.vf(0., p, args)

        def eq(x: RattleVars, args=None):
            C_q_1 = j_con_fun(x.q_1)
            zero = (p0 - step_size * 0.5 * ((dc.T @ x.lam) + dH_dq(x.p_1_2, q0)) - x.p_1_2,
                    q0 - step_size * 0.5 * (dH_dp(x.p_1_2, q0) + dH_dp(x.p_1_2, x.q_1)) - x.q_1, self.constrain(x.q_1),
                    x.p_1_2 - step_size * 0.5 * (dH_dq(x.p_1_2, x.q_1) + (C_q_1.T @ x.mu)) - x.p_1,
                    C_q_1 @ dH_dp(x.p_1, x.q_1))
            return zero

        def eq2(x: RattleVars, args=None):
            _, vjp_fun = jax.vjp(self.constrain, q0)
            _, vjp_fun_mu = jax.vjp(self.constrain, x.q_1)
            zero = (p0 - step_size * 0.5 * (vjp_fun(x.lam)[0] + dH_dq(x.p_1_2, q0)) - x.p_1_2,
                    q0 - step_size * 0.5 * (dH_dp(x.p_1_2, q0) + dH_dp(x.p_1_2, x.q_1)) - x.q_1, self.constrain(x.q_1),
                    x.p_1_2 - step_size * 0.5 * (dH_dq(x.p_1_2, x.q_1) + vjp_fun_mu(x.mu)[0]) - x.p_1,
                    jax.jvp(self.constrain, (x.q_1,), (dH_dp(x.p_1, x.q_1),))[1])
            return zero

        # term_1.vf_prod(0.0, q0, args, control1_half_1) == step_size * 0.5 *dH_dq(x.p_1_2, q0)
        # term_2.vf_prod(0., p0, args, control1_half_1) == step_size * 0.5 *dH_dp(p0, q0)
        def eq3(x: RattleVars, args=None):
            _, vjp_fun = jax.vjp(self.constrain, q0)
            _, vjp_fun_mu = jax.vjp(self.constrain, x.q_1)
            zero = (p0 - control1_half_1 * vjp_fun(x.lam)[0] - term_1.vf_prod(t0, q0, args, control1_half_1) - x.p_1_2,
                    q0 + term_2.vf_prod(t0, x.p_1_2, args, control2_half_1) + term_2.vf_prod(midpoint, x.p_1_2, args,
                                                                                             control2_half_2) - x.q_1,
                    self.constrain(x.q_1),
                    x.p_1_2 - term_1.vf_prod(midpoint, x.q_1, args, control1_half_2) - control1_half_2 *
                    vjp_fun_mu(x.mu)[0] - x.p_1, jax.jvp(self.constrain, (x.q_1,), (term_2.vf(t1, x.p_1, args),))[1])
            return zero

        def eq4(x: RattleVars, args=None):
            _, vjp_fun = jax.vjp(self.constrain, q0)
            _, vjp_fun_mu = jax.vjp(self.constrain, x.q_1)

            zero = ((p0 ** ω - control1_half_1 * (vjp_fun(x.lam)[0]) ** ω + term_1.vf_prod(t0, q0, args,
                                                                                           control1_half_1) ** ω - x.p_1_2 ** ω).ω,
                    (q0 ** ω + term_2.vf_prod(t0, x.p_1_2, args, control2_half_1) ** ω + term_2.vf_prod(midpoint,
                                                                                                        x.p_1_2, args,
                                                                                                        control2_half_2) ** ω - x.q_1 ** ω).ω,
                    self.constrain(x.q_1), (
                                x.p_1_2 ** ω + term_1.vf_prod(midpoint, x.q_1, args, control1_half_2) ** ω - (
                                    control1_half_2 * vjp_fun_mu(x.mu)[0] ** ω) - x.p_1 ** ω).ω,
                    jax.jvp(self.constrain, (x.q_1,), (term_2.vf(t1, x.p_1, args),))[1])
            return zero

        cs = jax.eval_shape(self.constrain, q0)

        init_vars = RattleVars(p_1_2=p0, q_1=(q0 ** ω * 2).ω, p_1=p0,
                               lam=jtu.tree_map(lambda cs: jnp.zeros(cs.shape, cs.dtype), cs),
                               mu=jtu.tree_map(lambda cs: jnp.zeros(cs.shape, cs.dtype), cs))


        sol = self.nonlinear_solver(eq4, init_vars, None)

        y1 = (sol.root.p_1, sol.root.q_1)
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(self, terms: Tuple[AbstractTerm, AbstractTerm], t0: Scalar, y0: Tuple[PyTree, PyTree], args: PyTree) -> \
    Tuple[PyTree, PyTree]:
        term_1, term_2 = terms
        y0_1, y0_2 = y0
        f1 = term_1.func(t0, y0_2, args)
        f2 = term_2.func(t0, y0_1, args)
        return (f1, f2)


if __name__ == '__main__':
    from diffrax import AbstractImplicitSolver, SaveAt, ODETerm, NewtonNonlinearSolver
    class Q(NamedTuple):
        x: Array
        y: Array


    def constrain(q: Q):
        return dict(c=jnp.sqrt(q.x ** 2 + q.y ** 2) - 1.) # check pytree constrain


    rat = Rattle(nonlinear_solver=NewtonNonlinearSolver(rtol=1e-4, atol=1e-6), constrain=constrain)


    def H(p: Q, q: Q):
        del q
        return (p.x ** 2 + p.y ** 2) / 2.

    def H(p: Q, q: Q):

        return (p.x ** 2 + p.y ** 2) / 2. + q.y

    terms = (ODETerm(lambda t, q, args: (-jax.grad(H, argnums=1)(jtu.tree_map(jnp.zeros_like, q), q) ** ω).ω),
             ODETerm(lambda t, p, args: jax.grad(H, argnums=0)(p, jtu.tree_map(jnp.zeros_like, p))))

    # p,q
    y0 = (Q(x=1., y=0.), Q(x=0., y=1.))
    y0 = jtu.tree_map(jnp.asarray, y0)
    t1 = 2 * jnp.pi / 4
    n = 2 ** 12
    dt = t1 / n
    saveat = SaveAt(t1=True, dense=True)


    solution = diffrax.diffeqsolve(terms, rat, 0.0, t1, dt0=dt, y0=y0, saveat=saveat)
    p1, q1 = solution.ys

    t = jnp.linspace(0.0, t1, 100)
    ps, qs = jax.vmap(solution.evaluate)(t)
    import matplotlib.pyplot as plt

    plt.plot(qs.x, qs.y)

    plt.gca().set_aspect('equal')
    plt.figure()
    plt.plot(t, jax.vmap(lambda q: jnp.sqrt(q.x**2+q.y**2))(ps) )
    plt.show()
