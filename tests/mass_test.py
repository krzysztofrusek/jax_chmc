from jax import config
config.update("jax_enable_x64", True)
import pytest
from scipy.sparse.linalg import eigsh

import numpy as np

from jax_chmc import Mass

'''
MATLAB:

co =

     8     1     6
     3     5     7
     4     9     2


M =

   102    71    53
    71    84    71
    53    71   102


dc =

     1     2     3

isinvM = false
cholM =

   10.0995         0         0
    7.0300    5.8803         0
    5.2478    5.8003    6.3888


Minv =

    0.0245   -0.0242    0.0041
   -0.0242    0.0528   -0.0242
    0.0041   -0.0242    0.0245


ans =

    3.8982

isinvM = true

cholM =

    0.1565         0         0
   -0.1544    0.1701         0
    0.0261   -0.1184    0.0990


Minv =

  102.0000   71.0000   53.0000
   71.0000   84.0000   71.0000
   53.0000   71.0000  102.0000


ans =

   -3.2314
'''




def test_mass_factor():
    M = np.asarray([[102, 71, 53],
                    [71, 84, 71],
                    [53, 71, 102]
                    ], dtype=np.float32)
    # isinvM = false
    m = Mass(M)
    expected = np.asarray([[10.0995, 0, 0],
                           [7.0300, 5.8803, 0],
                           [5.2478, 5.8003, 6.3888],
                           ])
    assert np.allclose(m.cholesky, expected)

    inv_expect = np.asarray([[0.0245, -0.0242, 0.0041],
                             [-0.0242, 0.0528, -0.0242],
                             [0.0041, -0.0242, 0.0245]
                             ])
    assert np.allclose(m.inverse, inv_expect, rtol=1e-3, atol=1e-4)

    dc = np.asarray([[1., 2, 3]])
    log_norm = m.compute_log_norm_const(dc)
    assert log_norm==pytest.approx(3.8982,0.05)

    D = dc @ m.inverse;
    cholMhat = m.cholesky - D.T @ np.linalg.solve(D @ D.T, D @ m.cholesky)

    s = eigsh(np.asarray(cholMhat @ cholMhat.T), k=dc.shape[1] - dc.shape[0], return_eigenvectors=False, which='LM')

    logNormConst = 0.5 * np.sum(np.log(s))

    assert logNormConst == pytest.approx(3.8982, 1e-4)

# def test_mass_factor_from_inv(self):
#     m = Mass(self.M)
#     expected = np.asarray([[0.1565, 0, 0],
#                            [-0.1544, 0.1701, 0],
#                            [0.0261, -0.1184, 0.0990],
#                            ])
#     col = m.cholesky
#     self.assertTrue(np.allclose(m.cholesky, expected))
#
#     inv_expect = np.asarray([[0.0245, -0.0242, 0.0041],
#                              [-0.0242, 0.0528, -0.0242],
#                              [0.0041, -0.0242, 0.0245]
#                              ])
#     self.assertTrue(np.allclose(m.inverse, inv_expect))


if __name__ == '__main__':
    pytest.main()
