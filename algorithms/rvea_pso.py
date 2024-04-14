from evox.operators import mutation, crossover, selection
from evox.operators.sampling import UniformSampling
from evox import Algorithm
from evox.utils import *


@jit_class
class RVEAPSO(Algorithm):
    """TensorRVEA algorithm variant with Particle Swarm Optimization operators.

    link: https://ieeexplore.ieee.org/document/7386636
          https://ieeexplore.ieee.org/document/494215

    Args:
        n_objs : The number of objectives.
        pop_size : The population size.
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
        inertia_weight=0.6,
        cognitive_coefficient=2.5,
        social_coefficient=0.8,
        mean=0,
        stdev=0.1,
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
        self.w = inertia_weight
        self.phi_p = cognitive_coefficient
        self.phi_g = social_coefficient
        self.mean = mean
        self.stdev = stdev

        self.selection = selection_op
        self.mutation = mutation_op
        self.crossover = crossover_op

        if self.selection is None:
            self.selection = selection.ReferenceVectorGuided()
        if self.mutation is None:
            self.mutation = mutation.Polynomial((lb, ub))
        if self.crossover is None:
            self.crossover = crossover.SimulatedBinary()

        self.sampling = UniformSampling(self.pop_size, self.n_objs)

    def setup(self, key):
        state_key, init_pop_key, init_v_key, sample_Key = jax.random.split(key, 4)

        v = self.sampling(sample_Key)[0]

        v0 = v
        self.pop_size = v.shape[0]
        if self.mean is not None and self.stdev is not None:
            population = self.stdev * jax.random.normal(
                init_pop_key, shape=(self.pop_size, self.dim)
            )
            population = jnp.clip(population, self.lb, self.ub)
            velocity = self.stdev * jax.random.normal(
                init_v_key, shape=(self.pop_size, self.dim)
            )
        else:
            length = self.ub - self.lb
            population = jax.random.uniform(
                init_pop_key, shape=(self.pop_size, self.dim)
            )
            population = population * length + self.lb
            velocity = jax.random.uniform(init_v_key, shape=(self.pop_size, self.dim))
            velocity = velocity * length * 2 - length


        return State(
            population=population,
            fitness=jnp.zeros((self.pop_size, self.n_objs)),
            velocity=velocity,
            local_best_location=population,
            local_best_fitness=jnp.full((self.pop_size,), jnp.inf),
            global_best_location=population[0],
            global_best_fitness=jnp.array([jnp.inf]),
            next_generation=population,
            reference_vector=v,
            init_v=v0,
            key=key,
            gen=0,
        )

    def init_ask(self, state):
        return state.population, state

    def init_tell(self, state, fitness):
        state = state.update(fitness=fitness)
        return state

    def ask(self, state):
        key, subkey, rg_key, rp_key, mut_key = jax.random.split(
            state.key, num=5
        )
        population = state.population

        no_nan_pop = ~jnp.isnan(population).all(axis=1)
        max_idx = jnp.sum(no_nan_pop).astype(int)
        pop = population[jnp.where(no_nan_pop, size=self.pop_size, fill_value=-1)]
        fitness = state.fitness[jnp.where(no_nan_pop, size=self.pop_size, fill_value=-1)]
        mating_pool = jax.random.randint(subkey, (self.pop_size,), 0, max_idx)
        population = pop[mating_pool]
        fitness = fitness[mating_pool]
        fitness = jnp.sum(fitness, axis=1)

        rg = jax.random.uniform(rg_key, shape=(self.pop_size, self.dim))
        rp = jax.random.uniform(rp_key, shape=(self.pop_size, self.dim))

        compare = state.local_best_fitness > fitness
        local_best_location = jnp.where(
            compare[:, jnp.newaxis], population, state.local_best_location
        )
        local_best_fitness = jnp.minimum(state.local_best_fitness, fitness)

        global_best_location, global_best_fitness = min_by(
            [state.global_best_location[jnp.newaxis, :], population],
            [state.global_best_fitness, fitness],
        )

        global_best_fitness = jnp.atleast_1d(global_best_fitness)

        velocity = (
                self.w * state.velocity
                + self.phi_p * rp * (local_best_location - population)
                + self.phi_g * rg * (global_best_location - population)
        )
        population = population + velocity

        next_generation = self.mutation(mut_key, population)
        next_generation = jnp.clip(next_generation, self.lb, self.ub)

        return next_generation, state.update(next_generation=next_generation, velocity=velocity,
                                                local_best_location=local_best_location,
                                                local_best_fitness=local_best_fitness,
                                                global_best_location=global_best_location,
                                                global_best_fitness=global_best_fitness, key=key)

    def tell(self, state, fitness):
        current_gen = state.gen + 1
        v = state.reference_vector
        merged_pop = jnp.concatenate([state.population, state.next_generation], axis=0)
        merged_fitness = jnp.concatenate([state.fitness, fitness], axis=0)

        survivor, survivor_fitness = self.selection(
            merged_pop, merged_fitness, v, (current_gen / self.max_gen) ** self.alpha
        )

        def rv_adaptation(pop_obj, v, v0):
            return v0 * (jnp.nanmax(pop_obj, axis=0) - jnp.nanmin(pop_obj, axis=0))

        def no_update(_pop_obj, v, v0):
            return v

        v = jax.lax.cond(
            current_gen % (1 / self.fr) == 0,
            rv_adaptation,
            no_update,
            survivor_fitness,
            v,
            state.init_v,
        )

        state = state.update(
            population=survivor,
            fitness=survivor_fitness,
            reference_vector=v,
            gen=current_gen,
        )
        return state
