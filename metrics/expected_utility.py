from jax import jit
import jax.numpy as jnp
from evox import jit_class
from typing import Callable
from functools import partial


@partial(jit, static_argnums=[2])
def eu(objs, w, utility_fun):
    return jnp.mean(jnp.max(utility_fun(objs, w.T), axis=0))


@jit_class
class ExpectedUtility:
    def __init__(self, w, utility_fun: Callable = jnp.dot):
        self.w = w
        self.utility_fun = utility_fun

    def __call__(self, objs):
        return eu(objs, self.w, self.utility_fun)