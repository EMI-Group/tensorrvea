from evox import problems, metrics
from evox.workflows import StdWorkflow
from algorithms import RVEAOrigin, TensorRVEA
from jax import random
import jax
import jax.numpy as jnp
import time
import json
from tqdm import tqdm
from evox.operators import non_dominated_sort


def run(algorithm_name, problem, key, num_iter=100, d=12):
    try:
        lb = jnp.zeros((d,))
        ub = jnp.ones((d,))

        if algorithm_name == "TensorRVEA":
            algorithm = TensorRVEA(lb=lb, ub=ub, n_objs=3, pop_size=105)
        elif algorithm_name == "RVEAOrigin":
            algorithm = RVEAOrigin(lb=lb, ub=ub, n_objs=3, pop_size=105)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        workflow = StdWorkflow(
            algorithm,
            problem,
        )
        state = workflow.init(key)
        step_func = jax.jit(workflow.step).lower(state).compile()
        state = step_func(state)
        run_time = []
        obj = []
        pop = []
        start = time.perf_counter()
        for k in range(num_iter):
            state = step_func(state)
            jax.block_until_ready(state)
            now = time.perf_counter()
            duration = now - start
            run_time.append(duration)
            obj.append(state.get_child_state("algorithm").fitness)
            pop.append(state.get_child_state("algorithm").population)
        return jnp.array(pop), jnp.array(obj), jnp.array(run_time)

    except Exception as e:
        import traceback
        traceback.print_stack()
        print("Error occurred:", e)
        return float("nan"), float("nan"), float("nan")


def evaluate(x, f, key, pf, alpha, num_iter=100):
    m = jnp.shape(pf)[1]
    ref = jnp.ones((m, ))
    igd = []
    hv = []
    ind1 = metrics.IGD(pf)
    ind2 = metrics.HV(ref=ref)
    history_data = []
    for i in range(num_iter):
        key, subkey = jax.random.split(key)
        current_pop = x[i]
        current_obj = f[i]
        current_obj = current_obj[~jnp.isnan(current_obj).any(axis=1)]
        current_pop = current_pop[~jnp.isnan(current_pop).any(axis=1)]
        rank = non_dominated_sort(current_obj)
        pf = rank == 0
        pf_fitness = current_obj[pf]
        pf_solutions = current_pop[pf]
        fmax = jnp.max(pf, axis=0)
        f_hv = pf_fitness / (alpha * fmax)
        f_hv = f_hv[jnp.all(f_hv <= 1, axis=1)]
        igd.append(ind1(pf_fitness))
        if f_hv.shape[0] > 0:
            hv.append(ind2(subkey, f_hv))
        else:
            hv.append(0)
        data = {"raw_pop": current_pop.tolist(), "raw_obj":current_obj.tolist(), "pf_solutions":pf_solutions.tolist(), "pf_fitness":pf_fitness.tolist()}
        history_data.append(data)

    return history_data, jnp.array(igd), jnp.array(hv)


if __name__ == "__main__":

    jax.config.update("jax_default_prng_impl", "rbg")
    num_iter = 100

    algorithm_names = ["TensorRVEA", "RVEAOrigin"]
    problem_list = [
            problems.numerical.DTLZ1(m=3),
            problems.numerical.DTLZ2(m=3),
            problems.numerical.DTLZ3(m=3),
            problems.numerical.DTLZ4(m=3),
            problems.numerical.DTLZ5(m=3),
            problems.numerical.DTLZ6(m=3),
            problems.numerical.DTLZ7(m=3),
    ]
    alpha_list = [1.5, 1.5, 50, 1.5, 5, 5, 5]
    num_runs = 31
    num_pro = 7

    experiment_stats = []
    key = random.PRNGKey(42)
    pro_keys = random.split(key, num_pro)

    for algorithm_name in algorithm_names:
        for j, problem in enumerate(problem_list):
            d = 7 if j == 0 else 12  # Dimension of the decision variables
            print(f"Running {algorithm_name} on DTLZ{j + 1} with dimension {d}")

            pro_key = pro_keys[j]
            run_keys = random.split(pro_key, num_runs)
            pf, _ = problem.pf(problem.init())
            for exp_id in tqdm(range(num_runs), desc=f"{algorithm_name} - Problem {j + 1}"):
                run_key = run_keys[exp_id]
                pop, obj, t = run(algorithm_name, problem, run_key, num_iter=num_iter, d=d)

                history_data, igd, hv = evaluate(
                    pop, obj, run_key, pf, alpha_list[j], num_iter=num_iter
                )

                data = {"history_data": history_data, "igd": igd.tolist(), "hv": hv.tolist(), "time": t.tolist()}
                with open(f"../data/model_performance/{algorithm_name}_DTLZ{j + 1}_exp{exp_id}.json", "w") as f:
                    json.dump(data, f)