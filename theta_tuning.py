"""Numerical calibration of the Counterbalance inflation parameter.

This module implements the matrix-agnostic bound used in the manuscript. It
maximizes the coupled bound B_theta(alpha, beta) over

    0 < alpha = rho - 1 < 6,
    0 < beta <= alpha * min(alpha, 1),

compares the result with the rho >= 7 chi-square branch, and numerically finds
theta such that the resulting bound is at most a requested delta.

The optimization is reproducible but numerical: it is not an interval-arithmetic
certificate of the global supremum. Use ``safety_margin`` (and optionally round
theta upward) when a conservative production constant is required.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import optimize, special


PAPER_THETA = {
    0.1: 1.40,
    0.05: 1.76,
    0.01: 3.00,
    0.001: 6.45,
}


@dataclass(frozen=True)
class CalibrationConfig:
    """Accuracy and reproducibility settings for the numerical search."""

    optimization_order: int = 48
    verification_orders: tuple[int, ...] = (96, 192)
    optimizer_seeds: tuple[int, ...] = (617, 9183)
    optimizer_tolerance: float = 3e-5
    optimizer_population: int = 10
    optimizer_iterations: int = 65
    alpha_min: float = 1e-8
    fraction_min: float = 1e-8
    theta_tolerance: float = 2e-4


@dataclass(frozen=True)
class BoundResult:
    """Result of maximizing the matrix-agnostic bound for a fixed theta."""

    theta: float
    value: float
    rho_at_max: float
    beta_at_max: float | None
    beta_fraction: float | None
    branch: str
    quadrature_values: tuple[float, ...]
    optimizer_success: bool


@dataclass(frozen=True)
class CalibrationResult:
    """Numerically calibrated theta and the bound evaluated at that theta."""

    delta: float
    target: float
    theta: float
    bound: BoundResult


@lru_cache(maxsize=None)
def _quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    if order < 8:
        raise ValueError("quadrature order must be at least 8")
    nodes, weights = leggauss(order)
    return (nodes + 1.0) / 2.0, weights / 2.0


def _chi1_cdf(x: np.ndarray | float) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    return special.erf(np.sqrt(np.maximum(values, 0.0) / 2.0))


def _chi1_pdf(x: np.ndarray | float) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    density = np.zeros_like(values)
    positive = values > 0.0
    density[positive] = (
        np.exp(-0.5 * values[positive])
        / np.sqrt(2.0 * math.pi * values[positive])
    )
    return density


def _chi4_cdf(x: float) -> float:
    if x <= 0.0:
        return 0.0
    return float(special.gammainc(2.0, 0.5 * x))


def _h_envelope(v: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Evaluate H_{alpha,beta}(v) from the manuscript."""

    values = np.asarray(v, dtype=float)
    result = np.zeros_like(values)
    result[values >= alpha] = 1.0

    interior = (values > 0.0) & (values < alpha)
    if np.any(interior):
        ratio = values[interior] / alpha
        chi_bound = _chi1_cdf(ratio)
        exponent = -alpha * alpha / (2.0 * beta) * (
            -np.log(ratio) - 1.0 + ratio
        )
        laplace_bound = np.exp(np.minimum(exponent, 0.0))
        result[interior] = np.minimum(chi_bound, laplace_bound)

    return result


def _z_h_convolution(
    y: np.ndarray,
    alpha: float,
    beta: float,
    quadrature_order: int,
) -> np.ndarray:
    """Compute integral_0^y p_chi1(z) H(y-z) dz.

    The substitution z = y*u^2 removes the chi-square density singularity at
    z=0. The integral is split where y*(1-u^2)=alpha, because H changes branch
    there.
    """

    y = np.asarray(y, dtype=float)
    if np.any(y <= 0.0):
        raise ValueError("convolution arguments must be positive")

    nodes, weights = _quadrature(quadrature_order)
    cutoff = np.sqrt(np.maximum(0.0, 1.0 - alpha / y))
    total = np.zeros_like(y)

    for lower, upper in (
        (np.zeros_like(cutoff), cutoff),
        (cutoff, np.ones_like(cutoff)),
    ):
        width = upper - lower
        u = lower[:, None] + width[:, None] * nodes[None, :]
        integrand = np.exp(-0.5 * y[:, None] * u * u) * _h_envelope(
            y[:, None] * (1.0 - u * u), alpha, beta
        )
        total += width * (integrand @ weights)

    return np.sqrt(2.0 * y / math.pi) * total


def coupled_bound(
    theta: float,
    alpha: float,
    beta: float,
    quadrature_order: int = 96,
) -> float:
    """Evaluate the manuscript's coupled bound B_theta(alpha, beta).

    ``alpha`` is the stable-rank offset ``rho - 1``. The admissible domain is
    ``0 < alpha < 6`` and ``0 < beta <= alpha * min(alpha, 1)``.
    """

    if theta <= 1.0:
        raise ValueError("theta must be greater than 1")
    if not 0.0 < alpha < 6.0:
        raise ValueError("alpha must satisfy 0 < alpha < 6")

    cap = alpha * min(alpha, 1.0)
    if not 0.0 < beta <= cap * (1.0 + 1e-12):
        raise ValueError("beta must satisfy 0 < beta <= alpha * min(alpha, 1)")
    beta = min(beta, cap)

    x = theta**-2
    m = min(alpha, 1.0)
    nodes, weights = _quadrature(quadrature_order)

    # Splitting at t=m resolves the change in G', while t=x-alpha resolves
    # the point at which H(y-z) can reach its v>=alpha branch.
    split_points = {0.0, x}
    for point in (m, x - alpha):
        if 0.0 < point < x:
            split_points.add(float(point))
    splits = sorted(split_points)

    total = 0.0
    for lower, upper in zip(splits[:-1], splits[1:]):
        # t = lower + (upper-lower)u^2 removes the square-root singularity
        # inherited from the chi-square density in G'.
        t = lower + (upper - lower) * nodes * nodes
        jacobian = 2.0 * (upper - lower) * nodes

        argument = (
            alpha * t - beta * np.minimum(t / m, 1.0)
        ) / (1.0 - t)
        derivative_numerator = np.where(
            t < m,
            alpha - beta / m,
            alpha - beta,
        )
        argument_derivative = derivative_numerator / (1.0 - t) ** 2

        g_derivative = np.zeros_like(t)
        valid = (argument > 0.0) & (argument_derivative > 0.0)
        g_derivative[valid] = (
            _chi1_pdf(argument[valid]) * argument_derivative[valid]
        )

        convolution = _z_h_convolution(
            x - t, alpha, beta, quadrature_order
        )
        total += float(
            np.dot(weights, jacobian * g_derivative * convolution)
        )

    return float(np.clip(total, 0.0, 1.0))


def _candidate_from_coordinates(
    coordinates: np.ndarray,
) -> tuple[float, float, float]:
    alpha = math.exp(float(coordinates[0]))
    fraction = math.exp(float(coordinates[1]))
    beta = fraction * alpha * min(alpha, 1.0)
    return alpha, beta, fraction


def _maximize_coupled_branch(
    theta: float,
    config: CalibrationConfig,
) -> BoundResult:
    lower_alpha = math.log(config.alpha_min)
    upper_alpha = math.log(6.0 * (1.0 - 1e-10))
    lower_fraction = math.log(config.fraction_min)
    upper_fraction = math.log(1.0 - config.fraction_min)

    candidates: list[tuple[float, float, float, bool]] = []
    for seed in config.optimizer_seeds:
        def objective(coordinates: np.ndarray) -> float:
            alpha, beta, _ = _candidate_from_coordinates(coordinates)
            return -coupled_bound(
                theta,
                alpha,
                beta,
                quadrature_order=config.optimization_order,
            )

        optimum = optimize.differential_evolution(
            objective,
            bounds=(
                (lower_alpha, upper_alpha),
                (lower_fraction, upper_fraction),
            ),
            seed=seed,
            tol=config.optimizer_tolerance,
            popsize=config.optimizer_population,
            maxiter=config.optimizer_iterations,
            polish=True,
            updating="immediate",
            workers=1,
        )
        alpha, beta, fraction = _candidate_from_coordinates(optimum.x)
        candidates.append((alpha, beta, fraction, bool(optimum.success)))

    best: BoundResult | None = None
    for alpha, beta, fraction, success in candidates:
        values = tuple(
            coupled_bound(theta, alpha, beta, order)
            for order in config.verification_orders
        )
        value = max(values)
        result = BoundResult(
            theta=float(theta),
            value=float(value),
            rho_at_max=float(1.0 + alpha),
            beta_at_max=float(beta),
            beta_fraction=float(fraction),
            branch="1 < rho < 7",
            quadrature_values=values,
            optimizer_success=success,
        )
        if best is None or result.value > best.value:
            best = result

    assert best is not None
    return best


def matrix_agnostic_bound(
    theta: float,
    config: CalibrationConfig = CalibrationConfig(),
) -> BoundResult:
    """Numerically maximize g(theta) over all branches in the manuscript."""

    if theta <= 1.0:
        raise ValueError("theta must be greater than 1")

    coupled = _maximize_coupled_branch(theta, config)
    large_rank_value = _chi4_cdf(theta**-2)
    if large_rank_value > coupled.value:
        return BoundResult(
            theta=float(theta),
            value=large_rank_value,
            rho_at_max=7.0,
            beta_at_max=None,
            beta_fraction=None,
            branch="rho >= 7",
            quadrature_values=(large_rank_value,),
            optimizer_success=True,
        )
    return coupled


def calibrate_theta(
    delta: float,
    config: CalibrationConfig = CalibrationConfig(),
    *,
    safety_margin: float = 0.0,
    round_up: float | None = None,
) -> CalibrationResult:
    """Find theta with numerical matrix-agnostic bound at most delta.

    ``safety_margin`` replaces the root target by
    ``delta * (1 - safety_margin)``. ``round_up`` can then round theta upward
    to a requested step, for example 0.01.
    """

    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if not 0.0 <= safety_margin < 1.0:
        raise ValueError("safety_margin must lie in [0, 1)")
    if round_up is not None and round_up <= 0.0:
        raise ValueError("round_up must be positive")

    target = delta * (1.0 - safety_margin)
    cache: dict[float, BoundResult] = {}

    def evaluate(theta: float) -> BoundResult:
        key = round(float(theta), 13)
        if key not in cache:
            cache[key] = matrix_agnostic_bound(float(theta), config)
        return cache[key]

    left = 1.0 + 1e-6
    left_result = evaluate(left)
    if left_result.value <= target:
        theta = left
    else:
        right = 2.0
        while evaluate(right).value > target:
            right *= 1.5
            if right > 1e8:
                raise RuntimeError("failed to bracket theta")

        theta = float(
            optimize.brentq(
                lambda current: evaluate(float(current)).value - target,
                left,
                right,
                xtol=config.theta_tolerance,
                rtol=config.theta_tolerance,
            )
        )

    if round_up is not None:
        theta = math.ceil(theta / round_up - 1e-12) * round_up

    bound = evaluate(theta)
    # Root-finding and global maximization are both numerical. If their noise
    # leaves the reported point just above the target, move theta upward until
    # the requested inequality is observed by the same verification search.
    increment = round_up or max(config.theta_tolerance, 1e-4)
    while bound.value > target:
        theta += increment
        if round_up is not None:
            theta = math.ceil(theta / round_up - 1e-12) * round_up
        bound = evaluate(theta)

    return CalibrationResult(
        delta=float(delta),
        target=float(target),
        theta=float(theta),
        bound=bound,
    )


def _quick_config() -> CalibrationConfig:
    return CalibrationConfig(
        optimization_order=32,
        verification_orders=(48, 96),
        optimizer_seeds=(617,),
        optimizer_tolerance=1e-4,
        optimizer_population=8,
        optimizer_iterations=45,
        theta_tolerance=5e-4,
    )


def _as_serializable(result: BoundResult | CalibrationResult) -> dict[str, object]:
    return asdict(result)


def _print_rows(rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    print("delta      theta       g(theta)    rho*        beta fraction")
    print("--------------------------------------------------------------")
    for row in rows:
        fraction = row["beta_fraction"]
        fraction_text = "--" if fraction is None else f"{float(fraction):.6f}"
        delta = row["delta"]
        delta_text = "--" if delta is None else f"{float(delta):g}"
        print(
            f"{delta_text:<10} "
            f"{float(row['theta']):<11.6f} "
            f"{float(row['value']):<11.8f} "
            f"{float(row['rho_at_max']):<11.6f} "
            f"{fraction_text}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Numerically calibrate the Counterbalance parameter theta."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--delta",
        nargs="+",
        type=float,
        help="failure probabilities to calibrate (default: 0.05)",
    )
    mode.add_argument(
        "--evaluate-theta",
        nargs="+",
        type=float,
        help="evaluate g(theta) without root-finding",
    )
    mode.add_argument(
        "--paper-table",
        action="store_true",
        help="reproduce the bound checks for the manuscript theta values",
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.0,
        help="solve against delta*(1-margin)",
    )
    parser.add_argument(
        "--round-up",
        type=float,
        default=None,
        help="round calibrated theta upward to this step",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use a faster, lower-accuracy search for development",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()

    config = _quick_config() if args.quick else CalibrationConfig()
    rows: list[dict[str, object]] = []

    if args.evaluate_theta:
        for theta in args.evaluate_theta:
            bound = matrix_agnostic_bound(theta, config)
            rows.append({"delta": None, **_as_serializable(bound)})
    elif args.paper_table:
        for delta, theta in PAPER_THETA.items():
            bound = matrix_agnostic_bound(theta, config)
            rows.append({"delta": delta, **_as_serializable(bound)})
    else:
        for delta in args.delta or [0.05]:
            calibrated = calibrate_theta(
                delta,
                config,
                safety_margin=args.safety_margin,
                round_up=args.round_up,
            )
            rows.append(
                {
                    "delta": calibrated.delta,
                    "target": calibrated.target,
                    "theta": calibrated.theta,
                    **_as_serializable(calibrated.bound),
                }
            )

    if args.json:
        print(json.dumps(rows, indent=2, allow_nan=False))
    else:
        _print_rows(rows)


if __name__ == "__main__":
    main()
