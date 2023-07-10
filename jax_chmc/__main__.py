import timeit

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from jax_chmc.kernels import fun_chmc


def make_chain(n=4):
    M = 3.0 * jnp.diag(jnp.ones(n))
    cm = fun_chmc(logdensity_fn=lambda q: -jnp.square(q).sum(),
                  sim_logdensity_fn=lambda q: -jnp.square(q).sum(),
                  con_fn=lambda q: q.sum(keepdims=True),
                  inverse_mass_matrix=M,
                  num_integration_steps=3,
                  step_size=0.3)
    return cm


def app():
    cm = make_chain()

    k = jax.random.PRNGKey(42)

    s0 = cm.init(jnp.zeros(4))
    s1, _ = cm.step(k, s0)

    ks = jax.random.split(k, 55048)
    _, qs = jax.lax.scan(lambda s, k: (cm.step(k, s)[0], s.position), s1, ks)

    sns.kdeplot(qs[5000:, :])
    plt.show()


def benchmark(few_large=True):
    """
    Apple M1 Pro  Sample mcmc: 0.14694534553905214 s ± 0.004017746250481866 s
        Lineax    Sample mcmc: 0.15854501789837627 s ± 0.0014483560794849987 s
        R**40     Sample mcmc: 0.20070723095312815 s ± 0.010753018901673742 s
        Li R**40  Sample mcmc: 0.2309575584294521 s ± 0.006085973709004624 s

    """
    n=40
    cm = make_chain(n)
    k = jax.random.PRNGKey(42)

    s0 = cm.init(jnp.zeros(n))
    s1, _ = cm.step(k, s0)

    if few_large:
        n_step = 256
        n = 4
    else:
        n_step = 2
        n = 4 * 128
    r = 16

    ks = jax.random.split(k, n_step)


    def sample():
        _, qs = jax.lax.scan(lambda s, k: (cm.step(k, s)[0], s.position), s1, ks)
        qs.block_until_ready()

    s = s0

    def _sample_py():
        s = s0
        for k in ks:
            s, q = cm.step(k, s)
            jax.block_until_ready(q)

    sample()
    t = timeit.repeat(sample, repeat=r, number=n, globals=dict(s=s))
    print(jax.devices())
    print(f'Sample mcmc: {np.mean(t) / r} s ± {2 * np.std(t) / np.sqrt(n * r)} s')


if __name__ == '__main__':
    benchmark()
