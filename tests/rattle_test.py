import unittest
from typing import NamedTuple

import diffrax
import matplotlib.pyplot as plt

from jax_chmc.rattle import Rattle
import jax.numpy as jnp
import jax
from diffrax import ODETerm, SaveAt
import numpy as np
from diffrax import NewtonNonlinearSolver
from diffrax.custom_types import Bool, DenseInfo, PyTree, Scalar,Array
import jax.tree_util as jtu
from equinox.internal import ω

class MyTestCase(unittest.TestCase):
    def test_something(self):
        def constrain(q):
            return jnp.sum(q, keepdims=True)
        rat = Rattle(constrain=constrain)
        def H(p,q):
            return p@p.T/4. + q@q.T


        # V = p^2/2m m=1, v=1


        terms = (ODETerm(lambda t, q, args: -jax.grad(H,argnums=1)(jnp.zeros_like(q), q) ),
                 ODETerm(lambda t, p, args: jax.grad(H, argnums=0)(p,jnp.zeros_like(p)))
                 )
        y0=(jnp.zeros((1)),jnp.ones((1)))
        state0 = rat.init(terms, 0.0, 0.01,y0,None)
        state1 = rat.step(terms, 0.0, 0.01,y0,None,state0,False)


        self.assertEqual(True, True)  # add assertion here

    def test_circle(self):
        def constrain(q):
            return jnp.sqrt(jnp.sum(q**2, keepdims=True))-1.

        rat = Rattle(nonlinear_solver=NewtonNonlinearSolver(rtol=1e-4, atol=1e-6),constrain=constrain)


        def H(p, q):
            del q
            return p @ p.T / 2.

        # V = p^2/2m m=1, v=1

        # (p,q)->(0,1)
        # dH/dp to funkcja p
        # dH/dq to funkcja q

        terms = (ODETerm(lambda t, q, args: -jax.grad(H, argnums=1)(jnp.zeros_like(q), q)),
                 ODETerm(lambda t, p, args: jax.grad(H, argnums=0)(p, jnp.zeros_like(p)))
                 )
        #p,q
        y0 = (jnp.asarray([1.,0.]), jnp.asarray([0.,1.]) )
        t1 = 2*jnp.pi/4
        n=2**12
        dt = t1/n
        saveat = SaveAt(t1=True)
        solution = diffrax.diffeqsolve(terms,rat,0.0,t1,dt0=dt,y0=y0,saveat=saveat)
        p1,q1 = solution.ys
        self.assertTrue( np.allclose(p1,jnp.asarray([0.,-1.]), rtol=1e-4, atol=1e-4) )
        self.assertTrue( np.allclose(q1, jnp.asarray([1., 0.]), rtol=1e-4, atol=1e-4) )


    def test_circle_pot(self):
        def constrain(q):
            return jnp.sqrt(jnp.sum(q**2, keepdims=True))-1.

        rat = Rattle(nonlinear_solver=NewtonNonlinearSolver(rtol=1e-4, atol=1e-6),constrain=constrain)


        def H(p, q):

            return p @ p.T / 2. + q[1]



        terms = (ODETerm(lambda t, q, args: -jax.grad(H, argnums=1)(jnp.zeros_like(q), q)),
                 ODETerm(lambda t, p, args: jax.grad(H, argnums=0)(p, jnp.zeros_like(p)))
                 )
        #p,q
        y0 = (jnp.asarray([1.,0.]), jnp.asarray([0.,1.]) )
        t1 = 2*jnp.pi/4
        n=2**12
        dt = t1/n
        saveat = SaveAt(t1=True,dense=True)
        solution = diffrax.diffeqsolve(terms,rat,0.0,t1,dt0=dt,y0=y0,saveat=saveat)
        p1,q1 = solution.ys


        t = jnp.linspace(0.0,t1,100)
        ps,qs = jax.vmap(solution.evaluate)(t)
        import matplotlib.pyplot as plt
        plt.plot(qs[:,0],qs[:,1])
        plt.plot(t, jax.vmap(jnp.linalg.norm)(ps))
        plt.gca().set_aspect('equal')
        plt.show()

    def test_circle_tree(self):

        class Q(NamedTuple):
            x:Array
            y:Array
        def constrain(q:Q):
            return jnp.sqrt(q.x**2+q.y**2)-1.

        rat = Rattle(nonlinear_solver=NewtonNonlinearSolver(rtol=1e-4, atol=1e-6),constrain=constrain)


        def H(p:Q, q:Q):
            del q
            return (p.x**2+p.y**2)/2.


        terms = (ODETerm(lambda t, q, args: (-jax.grad(H, argnums=1)(jtu.tree_map(jnp.zeros_like,q), q)**ω).ω),
                 ODETerm(lambda t, p, args: jax.grad(H, argnums=0)(p, jtu.tree_map(jnp.zeros_like,p)))
                 )


        #p,q
        y0 = (Q(x=1.,y=0.),Q(x=0.,y=1.))
        y0 = jtu.tree_map(jnp.asarray,y0)
        t1 = 2*jnp.pi/4
        n=2**12
        dt = t1/n
        saveat = SaveAt(t1=True, dense=True)


        #terms[0].vf(0.0, y0[0], None)

        solution = diffrax.diffeqsolve(terms,rat,0.0,t1,dt0=dt,y0=y0,saveat=saveat)
        p1,q1 = solution.ys
        self.assertAlmostEqual(p1.y, -1, places=4)

        t = jnp.linspace(0.0,t1,100)
        ps,qs = jax.vmap(solution.evaluate)(t)
        import matplotlib.pyplot as plt
        plt.plot(qs.x,qs.y)
        #plt.plot(t, jax.vmap(jnp.linalg.norm)(ps))
        plt.gca().set_aspect('equal')
        plt.show()



    def test_jacobians(self):

        def f(x):
            return jnp.sum(x, keepdims=True)

        Jf = jax.jacobian(f)
        x = jnp.asarray([1,2.])
        v=jnp.asarray([3.])
        J = Jf(x)


        y, vjp_fun = jax.vjp(f,x)
        jnp.allclose(v@J, vjp_fun(v)[0])

        v2= jnp.asarray([3., 4.])

        (primals_out, tangents_out) = jax.jvp(f,(x,), (v2,))
        jnp.allclose(tangents_out,J@v2)

if __name__ == '__main__':
    unittest.main()
