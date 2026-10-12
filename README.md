# Counterbalance theta calibration

This repository contains only the numerical calibration needed to choose the
Counterbalance inflation parameter `theta`. It does not contain the numerical
experiments from the paper.

The implementation follows the current coupled bound in the manuscript. For a
fixed `theta > 1`, it evaluates

```text
B_theta(alpha, beta),
alpha = rho - 1,
0 < alpha < 6,
0 < beta <= alpha * min(alpha, 1),
```

maximizes it over `(alpha, beta)`, compares it with the `rho >= 7`
chi-square branch, and then solves `g(theta) <= delta`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Usage

Calibrate the smallest numerical value for `delta = 0.05`:

```bash
python theta_tuning.py --delta 0.05
```

Round the result upward to two decimal places and retain a 5% margin in the
failure-probability target:

```bash
python theta_tuning.py --delta 0.05 --safety-margin 0.05 --round-up 0.01
```

Re-evaluate the conservative constants reported in the manuscript:

```bash
python theta_tuning.py --paper-table
```

The expected bound checks are approximately:

| `delta` | manuscript `theta` | `g(theta)` | maximizing `rho` |
|---:|---:|---:|---:|
| 0.1 | 1.40 | 0.094291 | 1.2391 |
| 0.05 | 1.76 | 0.045134 | 1.1628 |
| 0.01 | 3.00 | 0.008700 | 1.0576 |
| 0.001 | 6.45 | 0.000861 | 1.0126 |

Evaluate the bound at arbitrary parameter values without solving for a root:

```bash
python theta_tuning.py --evaluate-theta 1.4 1.76 3.0 6.45
```

Append `--json` for machine-readable output. `--quick` uses reduced optimizer
and quadrature settings and is intended only for development.

## Numerical status

The code uses transformed Gauss-Legendre quadrature for the two integrals and
reproducible differential-evolution searches for the supremum. Two optimizer
seeds and two refined quadrature orders are used by default. This is a
reproducible numerical maximization, not an interval-arithmetic certificate of
the global supremum. The `--safety-margin` and `--round-up` options provide a
practical numerical buffer; they do not turn the computation into a formal
proof.

Run the calibration-specific tests with:

```bash
python -m unittest discover -s tests -v
```
