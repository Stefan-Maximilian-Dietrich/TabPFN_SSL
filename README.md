![CI](https://github.com/Stefan-Maximilian-Dietrich/TabPFN_SSL/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Code Style](https://img.shields.io/badge/code%20style-ruff-blue)
[![codecov](https://codecov.io/gh/Stefan-Maximilian-Dietrich/TabPFN_SSL/branch/main/graph/badge.svg)](https://app.codecov.io/github/stefan-maximilian-dietrich/tabpfn_ssl)
# Bayesian Pseudo-Label Selection with Prior-Data Fitted Networks

**Project proposal & research repository**

Author: **Stefan Maximilian Dietrich**
Date: **October 2025**

---

## 📌 Table of Contents

* [Bayesian PLS in Semi-Supervised Learning](#bayesian-pls-in-semi-supervised-learning)
* [Prior-Data Fitted Networks (PFN)](#prior-data-fitted-networks-pfn)
* [Semi-Supervised Learning with PFNs](#semi-supervised-learning-with-pfns)
* [Related Work](#related-work)
* [Algorithmic Framework](#algorithmic-framework)
* [Test Settings](#test-settings)
* [Results](#results)
* [Disclaimer on the Use of LLMs](#disclaimer-on-the-use-of-llms)

---

## Bayesian PLS in Semi-Supervised Learning

Obtaining labeled data is often costly, time-consuming, and dependent on expert knowledge, whereas unlabeled data are typically abundant and easy to collect. This imbalance has led to the rise of **semi-supervised learning (SSL)**, with **self-training (or pseudo-labeling)** being one of the most widely used approaches.

Self-training iteratively adds pseudo-labeled instances to the training set based on predictions from a model trained on labeled data. A crucial step in this process is **pseudo-label selection (PLS)**, which determines which pseudo-labeled instances to include.
Importantly, *PLS refers to the selection of pseudo-labeled instances, not the pseudo-labels themselves.*

To mitigate overfitting and the reinforcement of incorrect pseudo-labels (confirmation bias), selection strategies should be less dependent on the current model and more informed by the structure and uncertainty inherent in the labeled dataset.

A Bayesian framework designed to reduce model dependency and improve robustness in the pseudo-label selection process was proposed in prior work and has since shown promising results and extensions.

---

### Pseudo Posterior Predictive (PPP)

The key idea is to avoid evaluating the likelihood of a pseudo-labeled instance
((x_i, \hat{y}_i)) under a single estimated parameter vector (\hat{\theta}), which can be prone to overfitting.

Instead, the **Pseudo Posterior Predictive (PPP)** criterion marginalizes over the *entire posterior distribution* of the model parameters (\theta):

[
p(D \cup (x_i, \hat{y}*i) \mid D)
= \int*\Theta p(D \cup (x_i, \hat{y}_i) \mid \theta), p(\theta \mid D), d\theta,
]

where (D) is the labeled dataset and (p(\theta \mid D)) is the posterior over parameters..

Intuitively, the PPP evaluates how well a candidate pseudo-labeled instance fits not just a single model but a distribution over plausible models.

This approach is both empirically motivated and theoretically grounded in **Bayesian decision theory**. Selecting the pseudo-labeled instance that maximizes the PPP corresponds to choosing the Bayes-optimal action under a utility function reflecting model fit.

---

### Scalability Limitations of Classical Bayesian Methods

While PPP-based methods can be approximated or computed directly for simple models, they do not scale to complex deep learning architectures. In high-dimensional parameter spaces, the required integrals become analytically and numerically intractable.

This motivates the use of **Prior-Data Fitted Networks (PFNs)**.

---

## Prior-Data Fitted Networks (PFN)

Prior-Data Fitted Networks (PFNs) approximate Bayesian inference by directly learning the **posterior predictive distribution (PPD)** of a given prior.

Formally, the PPD integrates over all possible hypotheses (\varphi):

[
p(y \mid x, D) \propto
\int_\Phi p(y \mid x, \varphi), p(D \mid \varphi), p(\varphi), d\varphi.
]

Instead of computing this integral explicitly (e.g. via MCMC), a transformer
(q_\theta(y \mid x, D)) is trained offline on synthetically generated datasets sampled from a known prior.

The PFN is optimized using:

[
\mathcal{L}*{\mathrm{PFN}} =
\mathbb{E}*{{(x_{\text{test}}, y_{\text{test}})} \cup D_{\text{train}}}
\left[

* \log q_\theta(y_{\text{test}} \mid x_{\text{test}}, D_{\text{train}})
  \right].
  ]

As a result, PFNs perform Bayesian inference **in a single forward pass**, without parameter updates at inference time.

---

### TabPFN

**TabPFN** is a PFN specialized for small tabular classification problems. It is trained on synthetic datasets generated from a prior combining Bayesian neural networks and structural causal models.

TabPFN:

* Performs in-context learning,
* Requires no hyperparameter tuning,
* Produces predictions in under a second,
* Achieves state-of-the-art performance on small tabular datasets.

This makes TabPFN a suitable tool for approximating the PPP.

---

## Semi-Supervised Learning with PFNs

A key theoretical result shows that:

[
p(D \cup (x_i, \hat{y}_i) \mid D)
= p(\hat{y}_i \mid x_i, D).
]

Thus, selection based on the posterior predictive distribution computed via TabPFN corresponds to the **Bayes-optimal action**.

This removes the need to explicitly specify a prior distribution and yields a **model-agnostic**, scalable pseudo-label selection strategy.

An interesting extension would involve defining multi-objective utility functions over multiple PFNs.

---

## Related Work

* Laplace-based Bayesian PLS methods rely on restrictive assumptions such as unimodality and low dimensionality.
* MCMC-based approaches are theoretically robust but computationally infeasible in SSL settings.
* Exact PPP computation is possible for conjugate models but limited in scope.
* Uncertainty-based pseudo-labeling methods often collapse uncertainty into a single metric.
* Existing TabPFN-based SSL approaches focus on ERM, whereas this work embeds TabPFN into a **decision-theoretic self-training loop** with iterative PPP recomputation.

---
## Installation

This repository relies on [uv](https://docs.astral.sh/uv/) for managing the virtual environment and project dependencies.  
Please ensure that `uv` is installed before continuing.

Follow the steps below to set up the project locally:

1. **Create a virtual environment**
   ```bash
   uv venv
   ```

2. **Activate the virtual environment**
   - On Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```

3. **Install the project in editable mode**
   ```bash
   uv pip install -e .
   ```

## Usage

The project can be executed in two different modes:

1. **Interactive experiment execution**
2. **Evaluation mode**

---

### Interactive Runner (default)

To start the interactive interface, run:

```bash
uv run python run_interactive.py
```

> **Note:** This is the default execution mode if the repository is not running inside the LRZ AI Cloud environment.

After launching the runner, you will see an overview of the current repository and execution mode:

```
======================================
 TabPFN_SSL - Interaktiver Runner
======================================
Repo:  <path-to-repository>
Mode:  local (auto_detect_lrz=False)
```

---

### 1. Select a Task Sheet

You will be prompted to choose one of the available task sheets:

```
Welches Task-Sheet möchtest du ausführen?
-> [1] Task_Spirals.py
   [2] local_smoke.py
   [3] toy_examples.py
Auswahl (1-3) [default 1]:
```

Enter the corresponding number (e.g. `3` for `toy_examples.py`).  
If no input is provided, the default option is selected.

> The structure of task sheets and the procedure for adding new experiments is explained in detail in the section  
> **Modules → Tasks** of this README.

---

### 2. Select Experiments

Next, specify which experiment(s) to execute:

```
Experiment wählen: Zahl (z.B. 3), Range (z.B. 0-10) oder 'all':
```

Supported formats:

- Single experiment: `3`
- Range of experiments: `0-10`
- All experiments: `all`

Example:
```
1-10
```

---

### 3. Configure Seeds

You will then be asked to define the seed configuration:

```
NUM_SEEDS [default 5]:
BASE_SEED [default 0]:
```

- `NUM_SEEDS` determines how many times an experiment is repeated.
- The seeds used range from `BASE_SEED` up to  
  `BASE_SEED + NUM_SEEDS - 1`.

For example:

- `BASE_SEED = 0`
- `NUM_SEEDS = 5`

→ Seeds `0, 1, 2, 3, 4` will be executed.

If no value is entered, the default values are used.

---

The selected experiments are then executed sequentially according to the chosen configuration.

---

### Evaluation Mode (placeholder)

The repository also provides an evaluation entry point:

```bash
uv run python run_evaluation.py
```

> Detailed documentation for evaluation mode will be added here.
## Algorithmic Framework

**Pseudo-Label Selection with PFN**

```
Input:
  Labeled data 𝓓
  Unlabeled data 𝓤

While stopping criterion not met:
  Fit classifier on 𝓓
  Predict pseudo-labels for all x ∈ 𝓤
  Compute PPP(x, ŷ(x)) using TabPFN
  Select a* = argmax PPP
  Update:
    𝓓 ← 𝓓 ∪ a*
    𝓤 ← 𝓤 \ a*

Output:
  Final fitted classifier
```

---
## Modular Architecture & Extensibility

This project is intentionally **modular**: you can extend or swap components without touching the rest of the system.  
The three core building blocks are:

1. **Datasets** (`data.py`) – provide data as a `pandas.DataFrame`
2. **Classifiers** (`classifier.py`) – provide a unified training/inference API
3. **Decision Rules** (`decision.py`) – decide *which* pseudo-labeled sample to select next (semi-supervised selection logic)

A key convention across the entire codebase is the **target column name**:

- The label column **must** be named: `target`
- All remaining columns are treated as features

---

## Datasets (`BaseDataset` + concrete dataset classes)

### Dataset contract
A dataset is a class inheriting from `BaseDataset` and implementing:

- `__init__(...)`: call `super().__init__(name="...")`
- `_load(self) -> pd.DataFrame`: return a DataFrame containing all features **and** a `target` column

`BaseDataset.__call__()` caches the loaded DataFrame and standardizes `target` by converting it to categorical codes `0,1,2,...` (and stores original categories in `df.attrs["target_categories"]`).

### Already implemented datasets
The following dataset classes are available out-of-the-box:

- `BreastCancer`
- `Iris`
- `Wine`
- `Bank` (Swiss banknotes subset)
- `MtcarsVS` (mtcars with `vs` as target)
- `Cassini` (synthetic 3-class, 2D)
- `Circle2D` (two circles)
- `Seeds` (UCI Seeds dataset)
- `Spirals` (two-spirals)

---

## Classifiers (unified training + prediction API)

### Classifier contract
A classifier is a class that implements:

- `__init__(...)`: set `self.name` and configure hyperparameters / underlying model
- `fit(self, df: pd.DataFrame, target_col: str = "target") -> self`
- `predict(self, data, target_col: str = "target") -> np.ndarray`
- `predict_proba(self, data, target_col: str = "target") -> np.ndarray`

Notes:
- `fit()` always expects a **DataFrame** with a `target` column.
- `predict()` / `predict_proba()` accept either a DataFrame (the `target` column is ignored if present) or array-like input.
- The `target_col` defaults to `"target"` everywhere for consistency.

### Already implemented classifiers
The following classifier wrappers are implemented:

- `TabPfnClassifier` (TabPFN v2 default)
- `NaiveBayesClassifier` (`variant="gaussian"` or `variant="multinomial"`)
- `MultinomialLogitClassifier` (logistic regression + standardization)
- `SmallNNClassifier` (MLP + standardization)
- `SVMClassifier` (RBF SVC + standardization, `probability=True`)
- `RandomForestCls`
- `GradientBoostingCls`
- `DecisionTreeCls`
- `KNNClassifier` (kNN + standardization)

---

## Decision Rules (pseudo-label selection logic)

Decision rules encapsulate **how to pick** the next pseudo-labeled point (or generally: which candidate to select) based on labeled and pseudo-labeled sets.

### Decision rule contract
A decision rule is a class that implements:

- `__init__(...)`: set `self.name` and optionally keep a classifier instance
- `__call__(self, labeled: pd.DataFrame, pseudo: pd.DataFrame) -> int`

where the return value is the **row index** (integer) of the selected sample within `pseudo`.

### Already implemented decision rules
The following selection rules exist:

- `maximalPPP`
- `SSL_prob`
- `SSL_confidence`

---

## How to extend the system

### 1) Add a new dataset
Create a new class in `data.py`:

- Inherit from `BaseDataset`
- Implement `_load()` returning a `pd.DataFrame`
- Ensure the label column is called **`target`**

Minimal skeleton:

```python
class MyDataset(BaseDataset):
    def __init__(self):
        super().__init__(name="MyDataset")

    def _load(self) -> pd.DataFrame:
        df = ...  # build / load your dataframe
        df["target"] = ...  # ensure target exists
        return df
## Test Settings

The proposed algorithm serves as a foundation for multiple method variants differing in:

* model architecture,
* stopping criteria.

Benchmarks include:

* supervised learning,
* SSL with ad-hoc selection strategies,
* soft revision methods,
* SLZ,
* TabPFN-D.

Experiments will be conducted on established tabular datasets under varying labeled/unlabeled ratios.

---

## Results

⬇️ **Place results here**

This section is intentionally left as a placeholder for:

* static preview plots (PNG),
* links to interactive plots hosted via GitHub Pages,
* quantitative result tables.

Example structure:

* `plots/…png` → shown below
* `docs/…html` → interactive version

---

## Disclaimer on the Use of LLMs

The use of large language models (LLMs) in the preparation of this work is outlined below:

* Development of research idea and proposal: **No LLMs used**
* Development of core content or substantive arguments: **No LLMs used**
* Improvement of language and writing style: **LLM used**
* Spelling and grammar checking: **LLM used**

---

*This repository represents an ongoing research project and will be extended with experimental results and code.*

## Citation

If you use this repository in your research, academic work, or derivative projects, please cite it as follows:

```
@misc{Dietrich2026TabPFN_SSL,
  author = {Stefan Maximilian Dietrich},
  title = {TabPFN_SSL: A Modular Framework for Semi-Supervised Learning with TabPFN},
  year = {2026},
}
```

If this project contributes to published work, proper citation is greatly appreciated.

---

## License

This project is distributed under the MIT License.

You are free to use, modify, and distribute this software in accordance with the terms specified in the LICENSE file.

For full details, see the [LICENSE](LICENSE) file included in this repository.