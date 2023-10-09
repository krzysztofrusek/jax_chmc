import unittest

from jax_chmc import mvn
import jax
import jax.numpy as jnp
import numpy as np

class MVNTestCase(unittest.TestCase):
    def test_static(self):
        cov = 0.5 * np.eye(4)
        loc = np.zeros(4)
        A = jnp.asarray([np.nan, np.nan, 1., 0])
        mubar, sigmabar = mvn.conditional_mvn(loc,cov,A)

        a = A[2:]
        jmubar, jsigmabar = mvn.static_conditional_mvn(loc,cov,a)
        self.assertTrue(np.allclose(jmubar, mubar))
        self.assertTrue(np.allclose(jsigmabar, sigmabar))

    def test_static(self):
        cov = 0.5 * np.eye(4)
        loc = np.zeros(4)
        A = jnp.asarray([1,1,0,0])
        b=jnp.asarray([0])
        #mubar, sigmabar = mvn.constrained_mvn(loc,cov,A)

        jmubar, jsigmabar = mvn.static_constrained_mvn(loc,cov,A,b)
        s = jax.random.multivariate_normal(jax.random.key(3),jmubar, jsigmabar,method='eigh')
        self.assertTrue(np.allclose(A@s,b))

        ...

if __name__ == '__main__':
    unittest.main()
