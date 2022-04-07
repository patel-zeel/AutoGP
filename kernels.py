import jax
import jax.numpy as jnp
from tinygp.helpers import dataclass, field, JAXArray
from tinygp import kernels

@dataclass
class Linear(kernels.Kernel):
    scale: JAXArray = field(default_factory=lambda: jnp.ones(()))
    sigma: JAXArray = field(default_factory=lambda: jnp.zeros(()))
    
    def evaluate(self, X1, X2):
        return (X1 / self.scale) @ (X2 / self.scale) + jnp.square(self.sigma)

# TODO: Add MLP kernel


def rbf(variance=1.0, scale=1.0, **kwargs): 
    kernel =  variance * kernels.ExpSquared(scale=scale, **kwargs)
    return jax.tree_map(jnp.array, kernel)


def periodic(variance=1.0, scale=1.0, gamma=1.0, **kwargs): 
    kernel =  variance * kernels.ExpSineSquared(scale=scale, gamma=gamma, **kwargs)
    return jax.tree_map(jnp.array, kernel)

def linear(scale=1.0, sigma=1.0, **kwargs): 
    kernel =  Linear(scale=scale, sigma=sigma, **kwargs)
    return jax.tree_map(jnp.array, kernel)

def exp(variance=1.0, scale=1.0,**kwargs): 
    kernel =  variance * kernels.Exp(scale=scale, **kwargs)
    return jax.tree_map(jnp.array, kernel)

def rat_quad(variance=1.0, alpha=1.0, **kwargs): 
    kernel =  variance * kernels.RationalQuadratic(alpha=alpha, **kwargs)
    return jax.tree_map(jnp.array, kernel)

def matern32(variance=1.0, scale=1.0, **kwargs): 
    kernel =  variance * kernels.Matern32(scale=scale, **kwargs)
    return jax.tree_map(jnp.array, kernel)

def matern52(variance=1.0, scale=1.0,**kwargs): 
    kernel =  variance * kernels.Matern52(scale=scale, **kwargs)
    return jax.tree_map(jnp.array, kernel)

def consine(variance=1.0, scale=1.0,**kwargs): 
    kernel =  variance * kernels.Cosine(scale=scale, **kwargs)
    return jax.tree_map(jnp.array, kernel)
    