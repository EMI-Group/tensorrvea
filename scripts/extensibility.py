import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
import time
import json
from tqdm import tqdm
from evox.utils import TreeAndVector
from evox.operators import non_dominated_sort
from evox.workflows import StdWorkflow
from problems import MoBrax
from algorithms import TensorRVEA, RVEACSO, RVEADE, RVEAPSO, RVEARandom
import os


def get_algorithm(algorithm_name, center, n_objs, pop_size):
    bounds = jnp.full_like(center, -8), jnp.full_like(center, 8)
    return {
        "RVEAGA": TensorRVEA(*bounds, n_objs=n_objs, pop_size=pop_size, uniform_init=False),
        "RVEACSO": RVEACSO(*bounds, n_objs=n_objs, pop_size=pop_size),
        "RVEADE": RVEADE(*bounds, n_objs=n_objs, pop_size=pop_size),
        "RVEAPSO": RVEAPSO(*bounds, n_objs=n_objs, pop_size=pop_size),
        "RVEARandom": RVEARandom(*bounds, n_objs=n_objs, pop_size=pop_size)
    }.get(algorithm_name, None)


# Model definition
class Model(nn.Module):
    act_shape: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(16)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.act_shape)(x)
        x = nn.tanh(x)
        return x


def run_workflow(algorithm, problem, key, adapter, num_iter):
    workflow = StdWorkflow(algorithm, problem, pop_transform=adapter.batched_to_tree, opt_direction="max")
    step_func = jax.jit(workflow.step)

    state = workflow.init(key)
    times, pops, objs = [], [], []
    for _ in range(num_iter):
        state = step_func(state)
        times.append(time.perf_counter())
        pops.append(state.get_child_state("algorithm").population)
        objs.append(-state.get_child_state("algorithm").fitness)
    return jnp.array(pops), jnp.array(objs), jnp.array(times)


def calculate_nd_history(pops, objs, num_iter):
    history_data = []
    for i in range(num_iter):
        current_pops = pops[i][~jnp.isnan(pops[i]).any(axis=1)]
        current_objs = -objs[i][~jnp.isnan(objs[i]).any(axis=1)]  # Negate to original objectives

        if i == 0:
            aggregated_pops, aggregated_objs = current_pops, current_objs
        else:
            aggregated_pops = jnp.concatenate([aggregated_pops, current_pops], axis=0)
            aggregated_objs = jnp.concatenate([aggregated_objs, current_objs], axis=0)

        ranks = non_dominated_sort(aggregated_objs)
        pf_mask = ranks == 0
        history_data.append({
            "pf_solutions": aggregated_pops[pf_mask].tolist(),
            "pf_fitness": (-aggregated_objs[pf_mask]).tolist()  # Return to maximization perspective
        })
    return history_data

def main():
    jax.config.update("jax_default_prng_impl", "rbg")
    num_iter, num_runs = 100, 10
    algorithm_list = ["RVEACSO", "RVEAGA", "RVEADE", "RVEAPSO", "RVEARandom"]
    envs = [
        {"name": "mo_hopper_m2", "observation_shape": 11, "action_shape": 3, "num_obj": 2},
        {"name": "mo_hopper_m3", "observation_shape": 11, "action_shape": 3, "num_obj": 3},
    ]

    directory = f"../data/extensibility_performance"
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    for algorithm_name in tqdm(algorithm_list, desc="Algorithms"):
        for env in tqdm(envs, desc="Environments"):
            for exp_id in range(num_runs):
                env_name = env["name"]
                model = Model(act_shape=env["action_shape"])
                key = random.PRNGKey(42)
                model_key, workflow_key = random.split(key)
                params = model.init(model_key, jnp.zeros((env["observation_shape"],)))
                adapter = TreeAndVector(params)
                center = adapter.to_vector(params)
                problem = MoBrax(policy=jax.jit(model.apply), env_name=env_name, cap_episode=1000, num_obj=env["num_obj"])

                algorithm = get_algorithm(algorithm_name, center, env["num_obj"], 10000)
                pops, objs, times = run_workflow(algorithm, problem, workflow_key, adapter, num_iter)
                history_data = calculate_nd_history(pops, objs, num_iter)

                data = {
                    "pf_solution_archive": history_data,
                    "final_pop": history_data[-1]["pf_solutions"],
                    "final_obj": history_data[-1]["pf_fitness"],
                    "time": times.tolist()
                }
                with open(f"{directory}/{algorithm_name}_{env_name}_exp{exp_id}.json", "w") as f:
                    json.dump(data, f)


if __name__ == "__main__":
    main()
