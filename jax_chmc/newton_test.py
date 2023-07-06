import unittest
from typing import NamedTuple

from jaxtyping import Float

import jax_chmc
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from jax_chmc.newton import newton_solve, newton_solver


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


    def test_newton_lineax(self):
        class X(NamedTuple):
            a: Float
            b: Float
        def ff(x: X, _):
            return dict(a=(x.a ** 2 - x.b + 1.),
                        b=(2 * x.a - x.b ** 2 + 1.))

        x0 = X(a=jnp.asarray([1.]), b=jnp.asarray([4.]))
        sol = newton_solve(ff, x0, max_iter=20)
        self.assertTrue(isinstance(sol.x,X))

        for l in jax.tree_util.tree_leaves(ff(sol.x,None) ):
            self.assertAlmostEqual(l,0,places=4)

    def test_newton_custom(self):
        class X(NamedTuple):
            a: Float
            b: Float
        def ff(x: X):
            return dict(a=(x.a ** 2 - x.b + 1.),
                        b=(2 * x.a - x.b ** 2 + 1.))

        x0 = X(a=jnp.asarray([1.]), b=jnp.asarray([4.]))
        sol = newton_solver(ff, x0, max_iter=20)
        self.assertTrue(isinstance(sol.x,X))

        for l in jax.tree_util.tree_leaves(ff(sol.x) ):
            self.assertAlmostEqual(l,0,places=4)

# add assertion here


if __name__ == '__main__':
    unittest.main()
