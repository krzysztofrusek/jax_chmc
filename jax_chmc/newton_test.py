import unittest

import jax_chmc
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

class NewtonTestCase(unittest.TestCase):
    def test_solver(self):
        f = lambda x: (x - 2.) * (x - 3)
        x0 = jnp.asarray([1.])

        sol = jax_chmc.newton.newton_solver(f, x0, max_iter=20, )
        self.assertAlmostEqual(float(sol.x), 2.)
        ...

    def test_solver_2d(self):
        f = lambda x: (x - 2.) * (x - 3)
        x0 = jnp.asarray([1., 4.])

        sol = jax_chmc.newton.newton_solver(jax.vmap(f), x0, max_iter=20, )
        self.assertTrue(jnp.allclose(sol.x, jnp.asarray([2, 3])))

    def test_vector_solver_2d(self):
        f = lambda x: (x - 2.) * (x - 3)
        x0 = jnp.asarray([1., 4.])

        sol = jax_chmc.newton.vector_newton_solver(jax.vmap(f), x0, max_iter=20, )
        self.assertTrue(jnp.allclose(sol.x, jnp.asarray([2, 3])))

    def test_norm_stop(self):
        f = lambda x: (x - 2.) * (x - 3)
        x0 = jnp.asarray([1.])

        sol = jax_chmc.newton.newton_solver(f, x0, max_iter=200, min_norm=1e-2)
        self.assertLess(int(sol.n), 200)

    def test_tree(self):
        f = lambda x: (x - 2.) * (x - 3)
        ff = lambda x: tree_map(f, x)
        x0 = dict(a=jnp.asarray([1.]), b=jnp.asarray([4.]))

        sol = jax_chmc.newton.newton_solver(ff, x0, max_iter=20)
        self.assertAlmostEqual(float(sol.x['a']), 2.)
        self.assertAlmostEqual(float(sol.x['b']), 3.)

    def test_aux(self):

        f = lambda x: ((x - 2.) * (x - 3),44)
        x0 = jnp.asarray([1.])

        sol = jax_chmc.newton.newton_solver(f, x0, max_iter=20,has_aux=True )
        self.assertAlmostEqual(float(sol.x), 2.)


# add assertion here


if __name__ == '__main__':
    unittest.main()
