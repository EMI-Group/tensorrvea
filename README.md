<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/evox_logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="./assets/evox_logo_light.png">
    <img alt="EvoX Logo" height="50" src="./assets/evox_logo_light.png">
  </picture>
  <br>
</h1>

<p align="center">
🌟 TensorRVEA: Tensorized Reference Vector Guided Evolutionary Algorithm 🌟
</p>

<p align="center">
  <a href="https://arxiv.org/">
    <img src="https://img.shields.io/badge/paper-arxiv-red?style=for-the-badge" alt="TensorRVEA Paper on arXiv">
  </a>
</p>

## Introduction
Tensorized RVEA (TensorRVEA) is proposed to enhance the scalability and efficiency of evolutionary multiobjective optimization by incorporating GPU acceleration. By adapting key data structures and operators into tensor forms, TensorRVEA seeks to utilize GPU-based parallel computing to offer a more efficient approach to complex optimization challenges.  The implementation of TensorRVEA is compatible with the [EvoX](https://github.com/EMI-Group/evox/) framewrok.

## Features
- **GPU Acceleration** 💻: Leverages GPUs for enhanced computational capabilities.
- **Large-Scale Optimization** 📈: Ideal for large population sizes and high-dimensional challenges.
- **Flexibility** 🔨: Compatible with a variety of tensor-based reproduction operators, including GA, DE, PSO, and CSO.
- **Real-World Applications** 🌐: Suited for complex tasks like multiobjective robotic control (MoBrax), with a special emphasis on neuroevolution methodologies.

## Requirements
TensorRVEA requires:
- EvoX (version >= 0.7.0)
- jax (version >= 0.4.16)
- jaxlib (version >= 0.3.0)
- brax (version >= 0.10.0)
- flax
- Visualization tools: plotly, pandas


## Example Usage
Sample example for DTLZ problems:

```python
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
```

Sample example for MoBrax:

```python
from algorithms import TensorRVEA
from evox.workflows import StdWorkflow
from evox.monitors import StdMOMonitor
from evox.utils import TreeAndVector
import jax
import jax.numpy as jnp
from flax import linen as nn
import time
import problems
from evox.operators.sampling import UniformSampling
from evox.metrics import HV
from metrics.expected_utility import ExpectedUtility

env_name = "mo_swimmer"


class Model(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(16)(x)
        x = nn.tanh(x)
        x = nn.Dense(2)(x)
        x = nn.tanh(x)
        return x


def main():
    key = jax.random.PRNGKey(43)
    model_key, workflow_key = jax.random.split(key)
    model = Model()
    params = model.init(model_key, jnp.zeros((8,)))
    adapter = TreeAndVector(params)
    monitor = StdMOMonitor(record_pf=False)

    problem = problems.MoBrax(
        policy=jax.jit(model.apply),
        env_name=env_name,
        cap_episode=1000,
        num_obj=2,
    )
    center = adapter.to_vector(params)

    workflow = StdWorkflow(
        algorithm=TensorRVEA(
            lb=jnp.full_like(center, -8),
            ub=jnp.full_like(center, 8),
            n_objs=2,
            pop_size=100,
            uniform_init=False,
        ),
        problem=problem,
        monitor=monitor,
        num_objectives=2,
        pop_transform=adapter.batched_to_tree,
        opt_direction="max",
    )

    state = workflow.init(workflow_key)
    step_func = jax.jit(workflow.step).lower(state).compile()
    state = step_func(state)
    w = UniformSampling(100, 2)()[0]
    ref = jnp.array([0, -1])
    hv_metric = HV(ref=-ref)
    eu_metric = ExpectedUtility(w=w)
    start = time.time()
    for i in range(100):
        key, subkey = jax.random.split(key)
        state = step_func(state)
        f = -state.get_child_state("algorithm").fitness
        f = f[~jnp.isnan(f).any(axis=1)]
        current_f = f[jnp.all(f >= ref, axis=1)]
        if current_f.shape[0] == 0:
            hv = 0
            eu = 0
        else:
            hv = hv_metric(jax.random.split(workflow_key)[1], -current_f)
            eu = eu_metric(current_f)
        print(f'Generation {i+1}, HV: {hv}, EU: {eu}')
    end = time.time()
    print(f"Total time: {end - start}s")


if __name__ == "__main__":
    main()
```

## Community & Support

- Engage in discussions and share your experiences on [GitHub Discussion Board](https://github.com/EMI-Group/evox/discussions).
- Join our QQ group (ID: 297969717).
- Help with the translation of the documentation on [Weblate](https://hosted.weblate.org/projects/evox/evox/).
We currently support translations in two languages, [English](https://evox.readthedocs.io/en/latest/) / [中文](https://evox.readthedocs.io/zh/latest/).
- Official Website: https://evox.group/
  
## Citing EvoX

If you use EvoX in your research and want to cite it in your work, please use:
```
@article{evox,
  title = {{EvoX}: {A} {Distributed} {GPU}-accelerated {Framework} for {Scalable} {Evolutionary} {Computation}},
  author = {Huang, Beichen and Cheng, Ran and Li, Zhuozhao and Jin, Yaochu and Tan, Kay Chen},
  journal = {arXiv preprint arXiv:2301.12457},
  eprint = {2301.12457},
  year = {2023}
}
