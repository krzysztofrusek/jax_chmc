import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

from jax_chmc.kernels import fun_chmc

if __name__ == '__main__':
    M = 3.0 * jnp.diag(jnp.ones(4))
    k = jax.random.PRNGKey(42)
    cm = fun_chmc(logdensity_fn=lambda q: -jnp.square(q).sum(),
                  sim_logdensity_fn=lambda q: -jnp.square(q).sum(),
                  con_fn=lambda q: q.sum(keepdims=True),
                  inverse_mass_matrix=M,
                  num_integration_steps=3,
                  step_size=0.3)
    s0 = cm.init(jnp.zeros(4))
    s1, _ = cm.step(k, s0)

    ks = jax.random.split(k, 55048)
    _, qs = jax.lax.scan(lambda s, k: (cm.step(k, s)[0], s.position), s1, ks)

    sns.kdeplot(qs[5000:, :])
    plt.show()
