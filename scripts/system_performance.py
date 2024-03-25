import os

# If you want to run on CPU, uncomment the following line
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from evox import problems
from evox.workflows import StdWorkflow
from algorithms import TensorRVEA, RVEAOrigin
from jax import random
import jax
import jax.numpy as jnp
import time
import numpy as np
import json
from tqdm import tqdm


def run(algorithm_name, problem, key, dim, pop_size, n_objs, num_iter=100):
    try:
        algorithm = {
            "TensorRVEA": TensorRVEA(lb=jnp.zeros((dim,)), ub=jnp.ones((dim,)), n_objs=n_objs, pop_size=pop_size),
            "RVEAOrigin": RVEAOrigin(lb=jnp.zeros((dim,)), ub=jnp.ones((dim,)), n_objs=n_objs, pop_size=pop_size)
        }.get(algorithm_name)

        if algorithm is None:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        workflow = StdWorkflow(algorithm, problem)
        state = workflow.init(key)
        step_func = jax.jit(workflow.step)
        start = time.perf_counter()
        for _ in range(num_iter):
            state = step_func(state)
            jax.block_until_ready(state)
        duration = time.perf_counter() - start
        return duration
    except Exception as e:
        print(f"Error running {algorithm_name} with dimension {dim}, pop_size {pop_size}: {e}")
        return float("nan")


if __name__ == "__main__":
    jax.config.update("jax_default_prng_impl", "rbg")
    num_iter = 100

    pop_scale_list = np.round(2 ** np.arange(5, 15)).astype(int) #5, 15
    dim_scale_list = np.round(2 ** np.arange(9, 19)).astype(int) #9, 19

    algorithm_list = ["RVEAOrigin", "TensorRVEA"]

    device = jax.default_backend()
    problem_list = [problems.numerical.DTLZ1(m=3)]

    num_runs = 31
    alg_keys = [random.PRNGKey(42), random.PRNGKey(43)]

    for i, algorithm_name in enumerate(algorithm_list):
        name = f"{algorithm_name}_DTLZ1_{device}"
        print(name)
        key = alg_keys[i]
        run_keys = random.split(key, num_runs)
        for exp_id in tqdm(range(num_runs), desc=f"Running {algorithm_name}"):
            pop_scale_durations = []
            dim_scale_durations = []
            key = run_keys[exp_id]

            # Collect pop_scale durations
            for pop_size in tqdm(pop_scale_list, desc="Pop size scaling"):
                duration = run(algorithm_name, problem_list[0], key, 100, pop_size, 3, num_iter)
                pop_scale_durations.append(duration)

            # Collect dim_scale durations
            for dim in tqdm(dim_scale_list, desc="Dimension scaling"):
                duration = run(algorithm_name, problem_list[0], key, dim, 100, 3, num_iter)
                if np.isnan(duration):
                    break
                dim_scale_durations.append(duration)

            # Organize data in the specified order
            data = {
                "pop_scale_list": pop_scale_list.tolist(),
                "pop_scale": pop_scale_durations,
                "dim_scale_list": dim_scale_list.tolist(),
                "dim_scale": dim_scale_durations,
            }

            # Save to JSON file
            with open(f"../data/system_performance/{name}_exp{exp_id}.json", "w") as f:
                json.dump(data, f)
