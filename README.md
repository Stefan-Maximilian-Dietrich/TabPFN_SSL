# Bayesian Pseudo-Label Selection with Prior-Data Fitted Networks (PFN-PLS)

This repository contains the **project proposal and ongoing research framework** for a novel semi-supervised learning (SSL) method that combines **Bayesian Pseudo-Label Selection (PLS)** with **Prior-Data Fitted Networks (PFNs)**, in particular **TabPFN**.

The goal is to develop a **decision-theoretically grounded, uncertainty-aware pseudo-label selection strategy** that scales beyond classical Bayesian models while remaining computationally feasible for modern tabular datasets.

---

## 📌 Table of Contents

* [Motivation](#motivation)
* [Core Idea](#core-idea)
* [Method Overview](#method-overview)
* [Algorithm](#algorithm)
* [Implementation](#implementation)
* [Results](#results)

  * [Semi-Synthetic Datasets](#semi-synthetic-datasets)
  * [Benchmark Comparisons](#benchmark-comparisons)
* [Planned Experiments](#planned-experiments)
* [Repository Status](#repository-status)
* [Disclaimer on LLM Use](#disclaimer-on-the-use-of-llms)
* [Author](#author)
* [References](#references)

---

## Motivation

Obtaining labeled data is often expensive, time-consuming, and dependent on expert knowledge, whereas unlabeled data are typically abundant. This imbalance has led to the widespread adoption of **semi-supervised learning**, particularly **self-training / pseudo-labeling** approaches.

A key challenge in self-training is **confirmation bias**: once incorrect pseudo-labels are added to the training set, errors tend to reinforce themselves. Many existing pseudo-label selection strategies rely on ad-hoc criteria such as prediction confidence or entropy, which are often overconfident and strongly dependent on a single fitted model.

This project addresses this issue by leveraging **Bayesian decision theory** and **posterior predictive uncertainty**.

---

## Core Idea

Instead of evaluating a pseudo-labeled sample ((x_i, \hat y_i)) under a single parameter estimate, we evaluate it under the **entire posterior distribution** using the **Pseudo Posterior Predictive (PPP)** criterion:

[
p(D \cup (x_i, \hat y_i) \mid D)
]

Under mild assumptions, this simplifies to:

[
p(D \cup (x_i, \hat y_i) \mid D) = p(\hat y_i \mid x_i, D)
]

This allows pseudo-label selection to be based directly on the **posterior predictive distribution (PPD)**.

### Key Insight

**Prior-Data Fitted Networks (PFNs)**, and specifically **TabPFN**, approximate the Bayesian posterior predictive distribution in a single forward pass.
Therefore, TabPFN can be used to efficiently approximate the PPP and select pseudo-labels in a **Bayes-optimal** manner.

---

## Method Overview

At each self-training iteration:

1. Train a classifier on the current labeled dataset.
2. Predict pseudo-labels for all unlabeled samples.
3. Compute the **PPP score** for each pseudo-labeled sample using TabPFN.
4. Select the sample with the highest PPP score.
5. Add the selected sample to the labeled set.
6. Repeat until a stopping criterion is met.

This approach is:

* **Model-agnostic**
* **Uncertainty-aware**
* **Grounded in Bayesian decision theory**
* **Computationally efficient**

---

## Algorithm

```
Input:
  D = labeled dataset
  U = unlabeled dataset

While stopping criterion not met:
  1. Fit classifier on D
  2. Predict pseudo-labels for U
  3. Compute PPP(x_i, ŷ_i) via TabPFN
  4. Select sample with maximal PPP
  5. Move selected sample from U to D

Output:
  Final trained classifier
```

---

## Implementation

The algorithm is designed to serve as a **generic framework**.
Different variants can be instantiated by changing:

* The base classifier,
* The stopping criterion,
* The utility function,
* The PFN / TabPFN configuration.

The repository will be extended with:

* Modular experiment pipelines,
* Reproducible benchmarks,
* Automated result aggregation.

---

## Results

This section illustrates **example result visualizations** in the style of comparable research repositories (e.g. *robust-pls*).
The images below are **placeholders** and should be replaced by actual experimental results.

### Semi-Synthetic Datasets

#### Banknote Dataset

| Labeled Samples         | Visualization                             |
| ----------------------- | ----------------------------------------- |
| n = 160 (80% unlabeled) | ![Banknote 160](results/banknote_160.png) |
| n = 120 (80% unlabeled) | ![Banknote 120](results/banknote_120.png) |
| n = 80  (80% unlabeled) | ![Banknote 80](results/banknote_80.png)   |

---

### Benchmark Comparisons

#### Mushrooms Dataset

| Labeled Samples | Visualization                               |
| --------------- | ------------------------------------------- |
| n = 120         | ![Mushrooms 120](results/mushrooms_120.png) |
| n = 160         | ![Mushrooms 160](results/mushrooms_160.png) |
| n = 200         | ![Mushrooms 200](results/mushrooms_200.png) |

#### Simulated Dataset

| Labeled Samples | Visualization                          |
| --------------- | -------------------------------------- |
| n = 60          | ![Simulation 60](results/sim_60.png)   |
| n = 100         | ![Simulation 100](results/sim_100.png) |
| n = 200         | ![Simulation 200](results/sim_200.png) |

---

## Planned Experiments

The proposed approach will be benchmarked against:

* Standard supervised learning
* SSL with ad-hoc selection strategies (confidence, entropy, variance)
* Soft Revision methods
* SLZ framework
* TabPFN-D

Experiments will systematically vary:

* Label ratios,
* Dataset sizes,
* Stopping criteria.

---

## Repository Status

🚧 **Research prototype / Work in progress**

This repository currently serves as:

* A research proposal,
* A conceptual reference implementation,
* A foundation for future experimental work.

---

## Disclaimer on the Use of LLMs

Large Language Models (LLMs) were used **only** for:

* Language polishing,
* Grammar and spelling checks.

LLMs were **not** used for:

* Developing the research idea,
* Designing the methodology,
* Formulating theoretical arguments.

---

## Author

**Stefan Maximilian Dietrich**
October 2025

---

## References

See the accompanying bibliography (`bib.bib`) for full citations.
