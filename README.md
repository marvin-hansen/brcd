# BRCD: Bayesian Root Cause Discovery

This repository contains the implementation of BRCD (Bayesian Root Cause Discovery), a novel approach for identifying root causes in causal systems using Bayesian inference. It also contains two folders to replicate the experiments in the paper.

## Environment Setup

To replicate the experiments in this repository, you need to read through the README files under each folder to setup the correct environment.

## Using BRCD with Your Own Data


### Prerequisites

- Python 3.8+
- Conda package manager

### Creating the Environment

1. Create a new conda environment:
```bash
conda create -n brcd python=3.8
conda activate brcd
```

2. Install core dependencies:
```bash
conda install numpy scipy pandas scikit-learn matplotlib
```

3. Install additional required packages:
```bash
pip install causal-learn==0.1.2.3
pip install pyagrum==0.21.0
pip install networkx==2.6.3
pip install tqdm==4.67.1
pip install igraph==0.11.8
pip install numba==0.61.0
pip install graphviz==0.19.1
pip install cairosvg==2.5.2
pip install pydot==1.4.2
pip install pydotplus==2.0.2
pip install pgmpy==0.1.19
pip install scikit-network==0.24.0
pip install graphical_models
pip install cliquepicking
```

4. Install additional dependencies for specific models:
```bash
# For AERCA
pip install requests==2.32.3

# For RCD
pip install python-igraph==0.9.9

# For CausalRCA
pip install requests==2.26.0
```

5. Verify installation by running:
```bash
python -c "import torch; import causal_learn; import pyagrum; print('Environment setup successful!')"
```


To run the BRCD algorithm on your own data, follow these steps:

### 1. Prepare Your Data

Ensure you have:
- `df_obs`: Observational data (normal conditions)
- `df_a`: Anomalous data (interventional conditions)
- Both datasets should be pandas DataFrames with the same column structure

### 2. Learn the Partial Causal Graph in the form of CPDAG

```python
from brcd import brcd_helper as brcd
from models.boss import boss
from graphical_models.classes.dags.pdag import PDAG
from models.epsilon_diagnosis import diagnose_boss_input
from utils import causal_learn_graph_to_nx_digraph

# Prepare column names
colnames = list(df_obs.columns)

# Learn CPDAG using BOSS algorithm
try:
    G_cl = boss(df_obs.to_numpy())
    arcs, edges = causal_learn_graph_to_nx_digraph(G_cl, colnames)
    cpdag = PDAG(nodes=colnames, arcs=arcs, edges=edges)
    X_clean_n_copy = df_obs.copy()
    X_clean_a_copy = df_a.copy()
except:
    # Handle zero variance features that may cause BOSS to fail
    report, X_float = diagnose_boss_input(df_obs, eps=1e-12)
    X_clean_n = df_obs.drop(columns=report["zero_variance_cols"])
    X_clean_a = df_a.drop(columns=report["zero_variance_cols"])
    X_clean_n_copy = X_clean_n.copy()
    X_clean_a_copy = X_clean_a.copy()
```

### 3. Run BRCD Algorithm

#### For Continuous Data (Single Root Cause)
```python
result = brcd(X_clean_n_copy, X_clean_a_copy,
                                        cpdag=cpdag,
                                        isdiscrete=False,
                                        num_root_causes_candidates = 1)
potential_root_causes = result['ranks']
print(potential_root_causes)
```

#### For Discrete Data (Single Root Cause)
```python
result = brcd(X_clean_n_copy, X_clean_a_copy,
                                        cpdag=cpdag,
                                        isdiscrete=True,
                                        num_root_causes_candidates = 1)
potential_root_causes = result['ranks']
print(potential_root_causes)
```

#### For Multiple Root Causes (k > 1)
```python
k = 2  # Specify the number of root causes to rank
result = brcd(X_clean_n_copy, X_clean_a_copy,
                                        cpdag=cpdag,
                                        isdiscrete=False,
                                        node_transform = "none",    
                                        transform_parents= True,
                                        num_root_causes_candidates = 2)
potential_root_causes = result['ranks']
print(potential_root_causes)
```

#### For bootstrapping version
```python
cpdag = None  # Specify the cpdag to be None
result = brcd(X_clean_n_copy, X_clean_a_copy,
                                        cpdag=cpdag,
                                        isdiscrete=False,
                                        num_root_causes_candidates = 1,
                                        bootstrap_samples = 10)
potential_root_causes = result['ranks']
print(potential_root_causes)
```


## Quick Example: Linear Gaussian Chain `X -> Y -> Z`

This example generates synthetic data from a linear Gaussian causal chain

```text
X -> Y -> Z
```

under normal conditions, then generates anomalous data by perturbing the mechanism (p(Y \mid X)). Since the conditional mechanism of `Y` changes, BRCD is expected to rank `Y` as the root cause.

Save the following script as `toy_x_y_z_brcd.py` in the **parent directory** of the cloned repository.

For example, if the repository is located at:

```bash
/Users/yourname/Desktop/brcd
```

then place the script at:

```bash
/Users/yourname/Desktop/toy_x_y_z_brcd.py
```

This is important because the package should be imported as:

```python
from brcd.brcd import brcd_helper as brcd
```

### Toy Example

```python
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

from graphical_models.classes.dags.pdag import PDAG
from brcd.brcd import brcd_helper as brcd


def generate_xyz_data(
    n=1000,
    seed=0,
    beta_xy=1.5,
    beta_yz=2.0,
    y_intercept=0.0,
    noise_x=1.0,
    noise_y=1.0,
    noise_z=1.0,
):
    """
    Generate data from the linear Gaussian SCM:

        X = eps_X
        Y = y_intercept + beta_xy * X + eps_Y
        Z = beta_yz * Y + eps_Z

    Changing y_intercept, beta_xy, or noise_y perturbs p(Y | X).
    """
    rng = np.random.default_rng(seed)

    X = rng.normal(0.0, noise_x, size=n)
    Y = y_intercept + beta_xy * X + rng.normal(0.0, noise_y, size=n)
    Z = beta_yz * Y + rng.normal(0.0, noise_z, size=n)

    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


def main():
    # ------------------------------------------------------------------
    # 1. Generate normal and anomalous data
    # ------------------------------------------------------------------

    df_obs = generate_xyz_data(
        n=1000,
        seed=1,
        beta_xy=1.5,
        beta_yz=2.0,
        y_intercept=0.0,
    )

    df_a = generate_xyz_data(
        n=1000,
        seed=2,
        beta_xy=1.5,
        beta_yz=2.0,
        y_intercept=2.0,  # perturb p(Y | X)
    )

    # ------------------------------------------------------------------
    # 2. Create the known CPDAG X - Y - Z
    # ------------------------------------------------------------------

    nodes = ["X", "Y", "Z"]

    cpdag = PDAG(
        nodes=nodes,
        arcs=[],
        edges=[("X", "Y"), ("Y", "Z")],
    )

    # ------------------------------------------------------------------
    # 3. Run BRCD
    # ------------------------------------------------------------------

    result = brcd(
        df_obs,
        df_a,
        cpdag=cpdag,
        isdiscrete=False,
        node_transform="none",
        transform_parents=True,
        num_root_causes_candidates=1,
    )

    print("\nFull BRCD result:")
    print(result)

    print("\nRanked potential root causes:")
    print(result["ranks"])


if __name__ == "__main__":
    main()
```

### Run the Example

From the parent directory of the cloned repository, run:

```bash
cd /Users/yourname/Desktop
conda activate brcd
python toy_x_y_z_brcd.py
```

The expected top-ranked root cause is `Y`, because the anomalous data was generated by changing the conditional mechanism (p(Y \mid X)). The variable `Z` may also change marginally, but its conditional mechanism (p(Z \mid Y)) remains unchanged.

### Expected Output

Since the anomalous data is generated by perturbing the mechanism \(p(Y \mid X)\), BRCD is expected to rank `Y` first:

```python
Full BRCD result:
{'ranks': ['Y', 'X', 'Z']}

Ranked potential root causes:
['Y', 'X', 'Z']


### Note on Multiprocessing

BRCD uses multiprocessing internally. Therefore, the BRCD call should be placed inside a function protected by:

```python
if __name__ == "__main__":
    main()
```

This avoids multiprocessing errors on macOS and other systems that use the `spawn` start method.
