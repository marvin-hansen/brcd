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
conda install numpy scipy pandas scikit-learn matplotlib seaborn
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


