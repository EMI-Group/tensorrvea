from evox.algorithms import NSGA2
from algorithms import TensorRVEA, MORandom
from evox.workflows import StdWorkflow
from problems import MoBrax
from jax import random
import jax
import jax.numpy as jnp
import time
from flax import linen as nn
import json
from tqdm import tqdm
from evox.utils import TreeAndVector
from evox.operators import non_dominated_sort


def get_algorithm(algorithm_name, center, n_objs, pop_size):
    bounds = jnp.full_like(center, -8), jnp.full_like(center, 8)
    return {
        "TensorRVEA": TensorRVEA(*bounds, n_objs=n_objs, pop_size=pop_size, uniform_init=False),
        "NSGAII": NSGA2(*bounds, n_objs=n_objs, pop_size=pop_size),
        "Random": MORandom(*bounds, n_objs=n_objs, pop_size=pop_size)
    }.get(algorithm_name, None)


def run_workflow(algorithm, problem, key, adapter, num_iter):
    workflow = StdWorkflow(algorithm, problem, pop_transform=adapter.batched_to_tree, opt_direction="max")
    state = workflow.init(key)
    step_func = jax.jit(workflow.step).lower(state).compile()
    times = []
    pop = []
    obj = []
    start = time.perf_counter()
    for _ in range(num_iter):
        state = step_func(state)
        times.append(time.perf_counter() - start)
        pop.append(state.get_child_state("algorithm").population)
        obj.append(-state.get_child_state("algorithm").fitness)

    return jnp.array(pop), jnp.array(obj), jnp.array(times)


def calculate_nd_history(pop, obj, num_iter):
    history_data = []

    for i in range(num_iter):

        current_pop = pop[i][~jnp.isnan(pop[i]).any(axis=1)]
        current_obj = -obj[i][~jnp.isnan(obj[i]).any(axis=1)]  # Negate back to original for ND sort

        if current_pop.ndim < 2 or current_obj.ndim < 2:
            print(
                f"Shape issue at iteration {i}: current_pop shape {current_pop.shape}, current_obj shape {current_obj.shape}")

        if i == 0:
            aggregated_pop = current_pop
            aggregated_obj = current_obj
        else:
            aggregated_pop = jnp.concatenate((aggregated_pop, current_pop), axis=0)
            aggregated_obj = jnp.concatenate((aggregated_obj, current_obj), axis=0)

        rank = non_dominated_sort(aggregated_obj)
        pf_indices = rank == 0
        pf_pop = aggregated_pop[pf_indices]
        pf_obj = aggregated_obj[pf_indices]

        history_data.append({"pf_solutions": pf_pop.tolist(), "pf_fitness": (-pf_obj).tolist()})

    return history_data


def main():
    jax.config.update("jax_default_prng_impl", "rbg")
    num_iter = 3
    algorithm_list = ["TensorRVEA", "NSGAII", "Random"]
    envs = [
        {"name": "mo_halfcheetah", "observation_shape": 17, "action_shape": 6, "num_obj": 2, "type": "continuous"},
        {"name": "mo_hopper_m2", "observation_shape": 11, "action_shape": 3, "num_obj": 2, "type": "continuous"},
        {"name": "mo_hopper_m3", "observation_shape": 11, "action_shape": 3, "num_obj": 3, "type": "continuous"},
        {"name": "mo_swimmer", "observation_shape": 8, "action_shape": 2, "num_obj": 2, "type": "continuous"},
    ]
    num_runs = 2
    alg_key = random.PRNGKey(42)

    for algorithm_name in tqdm(algorithm_list, desc="Algorithms"):
        for env in tqdm(envs, desc="Environments"):
            run_keys = random.split(alg_key, num_runs)
            for exp_id in range(num_runs):
                name = f"{algorithm_name}_{env['name']}_exp{exp_id}"
                model_key, workflow_key = jax.random.split(run_keys[exp_id])

                class PolicyModel(nn.Module):
                    @nn.compact
                    def __call__(self, x):
                        x = nn.Dense(16)(x)
                        x = nn.tanh(x)
                        x = nn.Dense(env["action_shape"])(x)
                        x = nn.tanh(x)
                        return x

                model = PolicyModel()
                params = model.init(model_key, jnp.zeros((env["observation_shape"],)))
                adapter = TreeAndVector(params)
                center = adapter.to_vector(params)
                problem = MoBrax(policy=jax.jit(model.apply), env_name=env["name"], cap_episode=1000,
                                     num_obj=env["num_obj"])

                algorithm = get_algorithm(algorithm_name, center, env["num_obj"], 100)
                if not algorithm:
                    raise ValueError(f"Algorithm {algorithm_name} not recognized")

                pops, objs, times = run_workflow(algorithm, problem, workflow_key, adapter, num_iter)
                history_data = calculate_nd_history(pops, objs, num_iter)

                raw_data = {
                    "pf_solution_archive": history_data,
                    "final_pop": history_data[-1]["pf_solutions"],
                    "final_obj": history_data[-1]["pf_fitness"],
                    "time": times.tolist()
                }

                with open(f"../data/multiobjective_neuroevolution/{name}.json", "w") as f:
                    json.dump(raw_data, f)



if __name__ == "__main__":
    main()
