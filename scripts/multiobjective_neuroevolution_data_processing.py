from jax import random
import jax.numpy as jnp
import numpy as np
import json
from tqdm import tqdm
from evox.operators.sampling import UniformSampling
from evox.metrics import HV
from metrics import ExpectedUtility
from evox.operators import non_dominated_sort
import os

def load_and_aggregate_data(algorithm, env_name, num_runs, num_iter, ref, w, metric_key):
    aggregated_pop, aggregated_obj, times = [], [], []
    hvs, eus = [], []

    hv_metric = HV(ref=(-1 * ref))
    eu_metric = ExpectedUtility(w=w)

    for exp_id in range(num_runs):
        hv, eu = [], []
        file_path = f"../data/multiobjective_neuroevolution/{algorithm}_{env_name}_exp{exp_id}.json"
        with open(file_path, "r") as file_data:
            data = json.load(file_data)

        pf_data = data["pf_solution_archive"]
        for k in range(num_iter):
            current_pf = jnp.array(pf_data[k]["pf_fitness"])
            current_pf = current_pf[jnp.all(current_pf >= ref, axis=1)]
            hv.append(hv_metric(random.split(metric_key)[1], -current_pf))
            eu.append(eu_metric(current_pf))

        aggregated_pop.append(jnp.array(data["final_pop"]))
        aggregated_obj.append(jnp.array(data["final_obj"]))
        times.append(data["time"])
        hvs.append(hv)
        eus.append(eu)

    aggregated_pop = jnp.vstack(aggregated_pop)
    aggregated_obj = jnp.vstack(aggregated_obj)
    _, unique_indices = np.unique(aggregated_pop, axis=0, return_index=True)
    aggregated_pop = aggregated_pop[np.sort(unique_indices)]
    aggregated_obj = aggregated_obj[np.sort(unique_indices)]
    times = jnp.array(times)

    return aggregated_pop, aggregated_obj, np.array(hvs), np.array(eus), times


def calculate_and_save_results(algorithm, env_name, pop, obj, hv, eu, times):
    rank = non_dominated_sort(-obj)
    pf = rank == 0
    pf_pop = pop[pf]
    pf_obj = obj[pf]

    metrics_directory = "../data/multiobjective_neuroevolution/metrics"
    pareto_directory = "../data/multiobjective_neuroevolution/pareto"

    if not os.path.exists(metrics_directory):
        os.makedirs(metrics_directory)

    if not os.path.exists(pareto_directory):
        os.makedirs(pareto_directory)

    # Saving HV, EU metrics, and computation times
    metrics_data = {"HV": hv.tolist(), "ExpectedUtility": eu.tolist(), "time": times.tolist()}
    metrics_file_path = f"{metrics_directory}/{algorithm}_{env_name}_metrics.json"
    with open(metrics_file_path, "w") as f:
        json.dump(metrics_data, f)

    # Saving Pareto optimal solutions
    pareto_data = {"pop": pf_pop.tolist(), "obj": pf_obj.tolist()}
    pareto_file_path = f"{pareto_directory}/{algorithm}_{env_name}_pareto.json"
    with open(pareto_file_path, "w") as f:
        json.dump(pareto_data, f)

    print(f"Saved metrics data to {metrics_file_path}")
    print(f"Saved Pareto data to {pareto_file_path}")


if __name__ == "__main__":
    num_iter, num_runs = 100, 10
    algorithm_list = ["TensorRVEA", "NSGAII", "Random"]
    envs = [
        {"name": "mo_halfcheetah", "observation_shape": 17, "action_shape": 6, "num_obj": 2, "type": "continuous", "ref": jnp.array([0, -599.78643799])},
        {"name": "mo_hopper_m2", "observation_shape": 11, "action_shape": 3, "num_obj": 2, "type": "continuous", "ref": jnp.array([0, -865.70227051])},
        {"name": "mo_hopper_m3", "observation_shape": 11, "action_shape": 3, "num_obj": 3, "type": "continuous",
         "ref": jnp.array([0, 0, -1942.84301758])},
        {"name": "mo_swimmer", "observation_shape": 8, "action_shape": 2, "num_obj": 2, "type": "continuous", "ref": jnp.array([0, -0.19898804])},
    ]

    metric_key = random.PRNGKey(42)

    for algorithm in tqdm(algorithm_list, desc="Algorithms"):
        for env in tqdm(envs, desc="Environments", leave=False):
            w = UniformSampling(105, 3)(metric_key)[0] if env["name"] == "mo_hopper_m3" else UniformSampling(100, 2)(metric_key)[0]
            aggregated_pop, aggregated_obj, hv, eu, times = load_and_aggregate_data(
                algorithm, env["name"], num_runs, num_iter, env["ref"], w, metric_key)

            calculate_and_save_results(algorithm, env["name"], aggregated_pop, aggregated_obj, hv, eu, times)
