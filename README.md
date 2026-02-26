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
## Theoretical Background

This repository builds on a decision-theoretic perspective on semi-supervised learning (SSL), combining Bayesian pseudo-label selection with Prior-Data Fitted Networks (PFNs), in particular TabPFN.

---

### 1) Bayesian Pseudo-Label Selection in SSL

Obtaining labeled data is often costly, time-consuming, and dependent on expert knowledge, whereas unlabeled data are typically abundant and easy to collect. This imbalance has led to the rise of semi-supervised learning (SSL), with self-training (or pseudo-labeling) being one of the most widely used approaches ([McClosky et al., 2006](https://aclanthology.org/N06-1020/))([Lee, 2013 (https://arxiv.org/abs/1303.7441))([Shi et al., 2018](https://arxiv.org/abs/1807.10308)). Self-training iteratively adds pseudo-labeled instances to the training set based on predictions from a model trained on labeled data. A crucial step in this process is pseudo-label selection (PLS), which determines which pseudo-labeled instances to include.\footnote{Importantly, PLS refers to the selection of pseudo-labeled instances, not the pseudo-labels themselves.}

To mitigate overfitting and the reinforcement of incorrect pseudo-labels (i.e., confirmation bias), selection strategies should be less dependent on the current model and more informed by the structure and uncertainty inherent in the labeled dataset. In~\cite{RodemannBPLS}, a Bayesian framework designed to reduce model dependency and improve robustness in the pseudo-label selection process was proposed. This framework has since shown promising results and extensions in~\cite{RodemannIAL, dietrich2024semisupervisedlearningguidedgeneralized, dietrich2024softrevisonBNN}.

This framework incorporates a principled approach for pseudo-label selection that explicitly accounts for model uncertainty: the \textit{Pseudo Posterior Predictive (PPP)}. This method provides a robust alternative to conventional selection strategies that rely solely on a single model’s predictions, which are often overconfident, especially in low-data or high-dimensional settings.

The key idea is to avoid evaluating the likelihood of a pseudo-labeled instance $(x_i, \hat{y}_i)$ under a single estimated parameter vector $\hat{\theta}$, which can be prone to overfitting. Instead, the PPP criterion marginalizes over the \textit{entire posterior distribution} of the model parameters $\theta$, thereby reflecting uncertainty about the correct parameterization. It is formally defined as:
\[
p(D \cup (x_i, \hat{y}_i) \mid D) = \int_\Theta p(D \cup (x_i, \hat{y}_i) \mid \theta) \, p(\theta \mid D) \, d\theta,
\]
where $D$ is the labeled dataset, and $p(\theta \mid D)$ is the posterior over parameters given the observed data. Intuitively, the PPP evaluates how well a candidate pseudo-labeled instance fits not just a single model but a distribution over plausible models.

This approach is both empirically motivated and theoretically grounded in Bayesian decision theory. \cite{RodemannBPLS} show that selecting the pseudo-labeled instance which maximizes the PPP corresponds to choosing the Bayes-optimal action under a utility function reflecting model fit. By doing so, the method naturally guards against early-stage confirmation bias and prioritizes examples that are consistently probable across the model’s posterior landscape.

In \cite{RodemannBPLS} and \cite{dietrich2024PLS}, it has been demonstrated that the integral in question can be feasibly approximated or, in certain simplified cases, computed directly. However, these methods have proven effective primarily for relatively simple model structures and datasets of very small to moderate size. They do not scale to more complex deep learning architectures, particularly when the parameter space $\theta$ resides in a high-dimensional domain, making the integral intractable both analytically and numerically. This is why we turn to the fruitful field of PFNs.

---

### 2) Prior-Data Fitted Networks and TabPFN

Prior-Data Fitted Networks (PFNs) approximate Bayesian inference directly using a neural network trained offline on many synthetic tasks sampled from a prior distribution.

Instead of performing expensive inference procedures such as MCMC or variational inference at test time, a PFN learns to approximate the posterior predictive distribution:

$$
p(y \mid x, D)
\propto
\int_\Phi
p(y \mid x, \phi)
\, p(D \mid \phi)
\, p(\phi)
\, d\phi
$$

A neural network $q_\theta(y \mid x, D)$ is trained over tasks $D \sim p(D)$ to approximate this predictive distribution.

**TabPFN** specializes this idea for small tabular classification problems. It performs in-context prediction without gradient-based retraining or hyperparameter tuning, while maintaining strong empirical performance in its target regime.

---

### 3) SSL with PFNs: Decision-Theoretic Selection via Posterior Predictives

A key theoretical insight is that, under a suitable formulation, the PPP selection criterion reduces to evaluating the posterior predictive probability:

$$
p(D \cup (x_i, \hat{y}_i) \mid D)
=
p(\hat{y}_i \mid x_i, D)
$$

This means that selecting the pseudo-label with maximal PPP corresponds to selecting the candidate with maximal posterior predictive probability — a Bayes-optimal decision under an appropriate utility formulation.

This connection makes PFNs particularly powerful in SSL:

- **Principled uncertainty awareness:** Selection is based on posterior predictive uncertainty rather than raw confidence scores.
- **Reduced confirmation bias:** Decisions integrate over plausible parameter configurations.
- **Computational efficiency:** TabPFN provides fast approximations of the posterior predictive without explicit Bayesian inference.
- **Modular design:** The pseudo-label selection mechanism can be separated from the downstream classifier.

By combining Bayesian pseudo-label selection with PFN-style posterior predictive modeling, this framework achieves a rare combination of strong theoretical grounding and practical efficiency — making it an attractive alternative to heuristic confidence- or entropy-based selection strategies in self-training pipelines.
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




Proper citation is appreciated if this project contributes to published or submitted work.
---



```
## Citation

If you use this repository in academic work, research projects, or derivative implementations, please cite it as follows:
```
@misc{Dietrich2026TabPFN_SSL,
  author = {Stefan Maximilian Dietrich},
  title  = {Bayesian Pseudo-Label Selection with Prior-Data Fitted Networks},
  year   = {2026},
}
```
---
## License

This project is licensed under the MIT License.

You are free to use, modify, and distribute this software in accordance with the terms specified in the LICENSE file.

For full details, see the [LICENSE](LICENSE) file included in this repository.
