from evox import workflows, problems
import algorithms
from evox.monitors import PopMonitor
from evox.metrics import IGD
import jax
import jax.numpy as jnp
import numpy as np
import time


def run_moea(algorithm, key):
    monitor = PopMonitor()

    problem = problems.numerical.DTLZ2(m=3)
    workflow = workflows.StdWorkflow(
        algorithm=algorithm,
        problem=problem,
        monitor=monitor,
    )

    state = workflow.init(key)

    true_pf, _ = problem.pf(problem.init())

    igd = IGD(true_pf)

    for i in range(100):
        key, subkey = jax.random.split(key)
        state = workflow.step(state)

        fit = state.get_child_state("algorithm").fitness
        non_nan_rows = fit[~np.isnan(fit).any(axis=1)]
        print(f'Generation {i+1}, IGD: {igd(non_nan_rows)}')
    fig = monitor.plot(state, problem)
    fig.show()


if __name__ == '__main__':
    lb = jnp.full(shape=(12,), fill_value=0)
    ub = jnp.full(shape=(12,), fill_value=1)

    algorithm = algorithms.TensorRVEA(
        lb=lb,
        ub=ub,
        n_objs=3,
        pop_size=100,
    )
    key = jax.random.PRNGKey(42)

    start = time.time()
    run_moea(algorithm, key)
    end = time.time()
    print(f"time: {end-start}s")
