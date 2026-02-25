![CI](https://github.com/Stefan-Maximilian-Dietrich/TabPFN_SSL/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Code Style](https://img.shields.io/badge/code%20style-ruff-blue)
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

where (D) is the labeled dataset and (p(\theta \mid D)) is the posterior over parameters.

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
