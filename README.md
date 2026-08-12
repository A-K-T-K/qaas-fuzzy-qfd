# QaaS Literature-Driven Fuzzy QFD Framework

Official Python implementation and multi-tiered sensitivity analysis engine for the research manuscript on **Quantum-as-a-Service (QaaS) platform capability prioritization**.

---

## Overview

This repository provides an auditable and reproducible decision-support framework that extends Quality Function Deployment (QFD) using Triangular Fuzzy Numbers (TFNs). The framework prioritizes service-layer software capabilities for cloud-based Quantum-as-a-Service (QaaS) platforms.

The execution engine translates literature-synthesized stakeholder requirements into prioritized architectural capabilities, then evaluates model stability through:

1. Fuzzy additive aggregation and centroid defuzzification.
2. Yager's total integral ranking index.
3. Deterministic crisp QFD benchmark comparison (1/3/9 scale).
4. Hurwicz optimism-index trajectory analysis ($\alpha$-cuts).
5. A multi-tiered diagnostic suite (Monte Carlo noise injection, parameter scaling, and TFN anchor perturbations).

---

## Repository Structure

```text
.
├── main.py                         # Computational engine, plotting, and diagnostics
├── requirements.txt                # Python dependencies
├── outputs/
│   └── figures/                    # Generated publication-ready vector graphics
│       ├── fig1_hoq_heatmap.pdf
│       ├── fig2_capability_ranking_bar.pdf
│       └── fig3_sensitivity_concurrent_boxplot.pdf
├── LICENSE                         # MIT License
└── README.md                       # Repository documentation
```

---

## Quickstart

### Prerequisites

- Python 3.8+
- `pip`

### Installation

```bash
git clone https://github.com/A-K-T-K/qaas-fuzzy-qfd.git
cd qaas-fuzzy-qfd
pip install -r requirements.txt
```

### Run the Analysis

Execute `main.py` to run the full computational pipeline, print summary tables to the terminal, execute sensitivity diagnostics, and generate all output figures:

```bash
python main.py
```

---

## Output Artifacts

Running `main.py` reproduces all numerical calculations and figures described in the manuscript:

- **House of Quality Matrix & Heatmap:** Generated as `outputs/figures/fig1_hoq_heatmap.pdf`.
- **Capability Ranking Bar Chart:** Generated as `outputs/figures/fig2_capability_ranking_bar.pdf`.
- **Concurrent Sensitivity Boxplot:** Generated as `outputs/figures/fig3_sensitivity_concurrent_boxplot.pdf`.
- **Terminal Diagnostics:** Full statistical execution output including Spearman correlation ($\rho$), Yager indices, and Monte Carlo stability metrics ($k \in \{1, 3, 5\}$).

---

## Dependencies

The required Python packages are listed in [`requirements.txt`](requirements.txt):

```text
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scipy>=1.7.0
tabulate>=0.8.9
```

---

## Citation & Metadata

Details will be updated upon final publication.

---

## License

This project is open-source software licensed under the [MIT License](LICENSE).