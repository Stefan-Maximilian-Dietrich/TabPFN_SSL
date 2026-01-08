# Bayesian Pseudo-Label Selection with Prior-Data Fitted Networks (PFN-PLS)

This repository contains the **project proposal and ongoing research code** for a novel semi-supervised learning (SSL) framework that combines **Bayesian Pseudo-Label Selection (PLS)** with **Prior-Data Fitted Networks (PFNs)**, in particular **TabPFN**.

The goal of this project is to develop a **decision-theoretically grounded, uncertainty-aware pseudo-label selection strategy** that scales beyond classical Bayesian models and remains computationally feasible for realistic tabular datasets.

---

## Motivation

Labeled data is often scarce, expensive, and time-consuming to obtain, while unlabeled data is abundant. Self-training and pseudo-labeling are popular SSL techniques, but they are highly sensitive to **confirmation bias**: once incorrect pseudo-labels are added, errors tend to reinforce themselves.

Classical pseudo-label selection strategies rely on:

* Maximum predicted probability,
* Predictive entropy,
* Predictive variance,

which are often **overconfident** and strongly tied to a single fitted model.

This project builds upon **Bayesian Pseudo-Label Selection (BPLS)** and replaces intractable Bayesian integrations with **Prior-Data Fitted Networks**, enabling robust pseudo-label selection even in complex settings.

---

## Core Idea

Instead of evaluating a pseudo-labeled sample ((x_i, \hat y_i)) under a single parameter estimate, we evaluate it under the **entire posterior distribution** using the **Pseudo Posterior Predictive (PPP)** criterion:

[
p(D \cup (x_i, \hat y_i) \mid D)
]

Under mild assumptions, this simplifies to:

[
p(D \cup (x_i, \hat y_i) \mid D) = p(\hat y_i \mid x_i, D),
]

which allows us to **directly use the posterior predictive distribution (PPD)** for selection.

### Key Insight

**TabPFN** approximates the Bayesian posterior predictive distribution in a single forward pass.
Therefore, TabPFN can be used to efficiently approximate the PPP and select pseudo-labeled samples in a **Bayes-optimal** way.

---

## Method Overview

At each self-training iteration:

1. Fit a classifier on the currently labeled dataset.
2. Predict pseudo-labels for all unlabeled samples.
3. Compute the **PPP score** for each pseudo-labeled sample using TabPFN.
4. Select the sample with the highest PPP score.
5. Treat it as ground truth and add it to the labeled set.
6. Repeat until a stopping criterion is met.

---

## Algorithm

**Pseudo Label Selection with PFN**

```
Input:
  Labeled data D
  Unlabeled data U

While stopping criterion not met:
  1. Fit a classifier on D
  2. Predict pseudo-labels for U
  3. Compute PPP(x_i, ŷ_i) using TabPFN
  4. Select sample with maximal PPP
  5. Move selected sample from U to D

Output:
  Final fitted classifier
```

This procedure is **model-agnostic**, scalable, and grounded in Bayesian decision theory.

---

## Relation to Prior Work

This project builds on and extends:

* Bayesian PLS via Laplace approximation (Rodemann et al.)
* Exact PPP computation for conjugate models
* Soft Revision methods
* TabPFN and TabPFN-D
* SSL via SLZ and uncertainty-aware pseudo-labeling

Unlike prior TabPFN-based SSL approaches, this framework:

* Uses TabPFN to approximate the **PPP**, not ERM,
* Recomputes PPP **iteratively**,
* Treats selected pseudo-labels as ground truth, influencing future selection.

---

## Planned Experiments

The proposed method will be evaluated against:

* Fully supervised learning
* Standard SSL with ad-hoc selection (confidence, entropy, variance)
* Soft Revision methods
* SLZ framework
* TabPFN-D

Experiments will be conducted on standard tabular datasets compatible with TabPFN, varying:

* Label ratios,
* Dataset sizes,
* Stopping criteria.

---

## Repository Status

🚧 **Work in progress / research prototype**

This repository currently serves as:

* A research proposal,
* A conceptual and algorithmic reference,
* A foundation for future experimental implementations.

Code, experiments, and results will be added incrementally.

---

## Disclaimer on the Use of LLMs

The use of Large Language Models (LLMs) in this project is limited to:

* Improving language and writing style
* Spelling and grammar checking

No LLMs were used for:

* Developing the research idea
* Designing the methodology
* Formulating theoretical arguments

---

## Author

**Stefan Maximilian Dietrich**
October 2025

---

## References

See the accompanying bibliography in the project proposal for full citations.
