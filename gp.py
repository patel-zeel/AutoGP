import jax
import jax.numpy as jnp
from tinygp import GaussianProcess

## Building GP
def build_gp(params, X):
    log_kernel, mean, log_noise = params
#     polynomial_fix(kernel)
    kernel = jax.tree_map(jnp.exp, log_kernel)  # Positivity transform
    noise = jnp.exp(log_noise)  # Positivity transform
    gp = GaussianProcess(kernel, X, diag=noise, mean=mean)
    return gp