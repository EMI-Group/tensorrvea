# --------------------------------------------------------------------------------------
# 1. RVEA algorithm is described in the following papers:
#
# Title: A Reference Vector Guided Evolutionary Algorithm for Many-Objective Optimization
# Link: https://ieeexplore.ieee.org/document/7386636
# --------------------------------------------------------------------------------------

import jax
import jax.numpy as jnp

from evox.operators import mutation, crossover
from evox.operators.sampling import UniformSampling
from evox import Algorithm, State, jit_class
from jax import jit, lax
from functools import partial


@partial(jit, static_argnames="func")
def pairwise_func(x, y, func):
    return jax.vmap(lambda _x: jax.vmap(lambda _y: func(_x, _y))(y))(x)

@jit
def _cos_dist(x, y):
    return jnp.dot(x, y) / (jnp.linalg.norm(x) * jnp.linalg.norm(y))

@jit
def cos_dist(x, y):
    return pairwise_func(x, y, _cos_dist)

@jit
def ref_vec_guided(x, f, v, theta):
    n, m = jnp.shape(x)
    nv = jnp.shape(v)[0]

    obj = f - jnp.nanmin(f, axis=0)
    obj = jnp.maximum(obj, 1e-32)

    cosine = cos_dist(v, v)

    cosine = jnp.where(jnp.eye(jnp.shape(cosine)[0], dtype=bool), 0, cosine)
    cosine = jnp.clip(cosine, 0, 1)
    gamma = jnp.min(jnp.arccos(cosine), axis=1)

    angle = jnp.arccos(jnp.clip(cos_dist(obj, v), 0, 1))

    nan_mask = jnp.isnan(obj).any(axis=1)
    associate = jnp.argmin(angle, axis=1)
    associate = jnp.where(nan_mask, -1, associate)

    next_ind = jnp.full(nv, -1)
    is_null = jnp.sum(next_ind)

    vals = next_ind

    def update_next(i, sub_index, next_ind):
        apd = (1 + m * theta * angle[sub_index, i] / gamma[i]) * jnp.sqrt(
            jnp.sum(obj[sub_index, :] ** 2, axis=1)
        )

        apd_max = jnp.max(apd)
        noise = jnp.where(sub_index == -1, apd_max, 0)
        apd = apd + noise
        best = jnp.argmin(apd)

        next_ind = next_ind.at[i].set(sub_index[best.astype(int)])
        return next_ind

    def no_update(i, sub_index, next_ind):
        return next_ind

    def body_fun(i, vals):
        next_ind = vals
        sub_index = jnp.where(associate == i, size=nv, fill_value=-1)[0]

        next_ind = lax.cond(
            jnp.sum(sub_index) != is_null,
            update_next,
            no_update,
            i,
            sub_index,
            next_ind,
        )
        return next_ind

    next_ind = lax.fori_loop(0, nv, body_fun, vals)

    mask3 = next_ind == -1
    next_x = jnp.where(mask3[:, jnp.newaxis], jnp.nan, x[next_ind])
    next_f = jnp.where(mask3[:, jnp.newaxis], jnp.nan, f[next_ind])

    return next_x, next_f

@jit_class
class ReferenceVectorGuided:
    """Reference vector guided environmental selection."""

    def __call__(self, x, f, v, theta):
        return ref_vec_guided(x, f, v, theta)


@jit_class
class RVEAOrigin(Algorithm):
    """Origin RVEA algorithms

    link: https://ieeexplore.ieee.org/document/7386636

    Args:
        alpha : The parameter controlling the rate of change of penalty. Defaults to 2.
        fr : The frequency of reference vector adaptation. Defaults to 0.1.
        max_gen : The maximum number of generations. Defaults to 100.
        If the number of iterations is not 100, change the value based on the actual value.
        uniform_init : Whether to initialize the population uniformly. Defaults to True.
    """

    def __init__(
        self,
        lb,
        ub,
        n_objs,
        pop_size,
        alpha=2,
        fr=0.1,
        max_gen=100,
        uniform_init=True,
        selection_op=None,
        mutation_op=None,
        crossover_op=None,
    ):
        self.lb = lb
        self.ub = ub
        self.n_objs = n_objs
        self.dim = lb.shape[0]
        self.pop_size = pop_size
        self.alpha = alpha
        self.fr = fr
        self.max_gen = max_gen
        self.uniform_init = uniform_init

        self.selection = selection_op
        self.mutation = mutation_op
        self.crossover = crossover_op

        if self.selection is None:
            self.selection = ReferenceVectorGuided()
        if self.mutation is None:
            self.mutation = mutation.Polynomial((lb, ub))
        if self.crossover is None:
            self.crossover = crossover.SimulatedBinary()

        self.sampling = UniformSampling(self.pop_size, self.n_objs)

    def setup(self, key):
        key, subkey1, subkey2 = jax.random.split(key, 3)

        v = self.sampling(subkey2)[0]
        v0 = v
        self.pop_size = v.shape[0]

        if self.uniform_init:
            population = (
                    jax.random.uniform(subkey1, shape=(self.pop_size, self.dim))
                    * (self.ub - self.lb)
                    + self.lb
            )
        else:
            population = (
                    jax.random.normal(subkey1, shape=(self.pop_size, self.dim)) * 0.1
            )


        return State(
            population=population,
            fitness=jnp.zeros((self.pop_size, self.n_objs)),
            next_generation=population,
            reference_vector=v,
            init_v=v0,
            is_init=True,
            key=key,
            gen=0,
        )

    def init_ask(self, state):
        return state.population, state

    def init_tell(self, state, fitness):
        state = state.update(fitness=fitness)
        return state

    def ask(self, state):
        key, subkey, x_key, mut_key = jax.random.split(state.key, 4)
        population = state.population

        no_nan_pop = ~jnp.isnan(population).all(axis=1)
        max_idx = jnp.sum(no_nan_pop).astype(int)

        pop = population[jnp.where(no_nan_pop, size=self.pop_size, fill_value=-1)]

        mating_pool = jax.random.randint(subkey, (self.pop_size,), 0, max_idx)
        crossovered = self.crossover(x_key, pop[mating_pool])
        next_generation = self.mutation(mut_key, crossovered)

        return next_generation, state.update(next_generation=next_generation, key=key)

    def tell(self, state, fitness):
        current_gen = state.gen + 1
        v = state.reference_vector
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_fitness = jnp.concatenate([state.fitness, fitness], axis=0)

        survivor, survivor_fitness = self.selection(
            merged_pop, merged_fitness, v, (current_gen / self.max_gen) ** self.alpha
        )

        def rv_adaptation(pop_obj, v):
            v_temp = v * jnp.tile((jnp.nanmax(pop_obj, axis=0) - jnp.nanmin(pop_obj, axis=0)), (len(v), 1))

            next_v = v_temp / jnp.tile(
                jnp.sqrt(jnp.sum(v_temp ** 2, axis=1)).reshape(len(v), 1),
                (1, jnp.shape(v)[1]),
            )

            return next_v

        def no_update(_pop_obj, v):
            return v

        v = jax.lax.cond(
            current_gen % (1 / self.fr) == 0,
            rv_adaptation,
            no_update,
            survivor_fitness,
            state.init_v,
        )

        state = state.update(
            population=survivor,
            fitness=survivor_fitness,
            reference_vector=v,
            gen=current_gen,
        )
        return state
