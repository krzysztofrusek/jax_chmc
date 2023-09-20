import unittest

import jax.numpy as jnp
import jax.random
import numpy as np

from jax_chmc import mvn
from jax_chmc import Mass
from jax_chmc import fun_chmc


class FunCHMCCase(unittest.TestCase):
    def test_something(self):
        logdens = lambda x: -(x ** 2).sum()
        condfun = lambda x: x.sum(keepdims=True)
        M = 3.0 * jnp.diag(jnp.ones(4))
        c = fun_chmc(logdensity_fn=logdens,
                     sim_logdensity_fn=logdens,
                     con_fn=condfun,
                     step_size=0.1,
                     inverse_mass_matrix=M,
                     num_integration_steps=2
                     )

        s0 = c.init(jnp.ones(4))
        k = jax.random.PRNGKey(52)
        s1 = c.step(k, s0)
        ...

    def test_mvn(self):
        M = 3.0 * jnp.diag(jnp.ones(4))
        k = jax.random.PRNGKey(42)

        init_fn, step_fn = fun_chmc(logdensity_fn=lambda q: -jnp.square(q).sum(),
                                    sim_logdensity_fn=lambda q: -jnp.square(q).sum(),
                                    con_fn=lambda q: q[:2].sum(keepdims=True),
                                    step_size=0.1,
                                    inverse_mass_matrix=M,
                                    num_integration_steps=2
                                    )

        s0 = init_fn(jnp.zeros(4))

        s1, _ = step_fn(k, s0)

        ks = jax.random.split(k, 55048)

        def step(state, k):
            new_state, info = step_fn(k, state)
            return new_state, (new_state, info)

        state, (states, infos) = jax.lax.scan(step, s1, ks)

        cov = 0.5 * np.eye(s0.position.shape[0])
        loc = np.zeros((4, 1))
        A = jnp.asarray([1, 1, 0, 0])
        mubar, sigmabar = mvn.constrained_mvn(loc, cov, A)
        qs = states.position
        sampledsigma = np.cov(qs[2000:, :].T)
        self.assertTrue(np.allclose(sampledsigma, sigmabar, atol=0.05))


class MassCase(unittest.TestCase):

    def test_mass(self):
        M = 3.1 * jnp.diag(jnp.ones(4))
        m = Mass(M)


if __name__ == '__main__':
    unittest.main()
