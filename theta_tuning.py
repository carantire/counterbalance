from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import optimize, special


@dataclass(frozen=True)
class CalibrationConfig:
    rho_max: float = 7.0
    rho_grid_size: int = 800
    quad_order: int = 256
    xatol: float = 1e-7


class CounterbalanceCalibrator:
    def __init__(self, config: CalibrationConfig = CalibrationConfig()) -> None:
        self.config = config
        self.nodes, self.weights = leggauss(config.quad_order)

    def integrate(self, f: Callable[[np.ndarray], np.ndarray], a: float, b: float) -> float:
        if b <= a:
            return 0.0

        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        points = mid + half * self.nodes

        return float(half * np.dot(self.weights, f(points)))

    @staticmethod
    def chi1_cdf(x: np.ndarray | float) -> np.ndarray:
        x = np.asarray(x)
        return special.erf(np.sqrt(np.maximum(x, 0.0) / 2.0))

    @staticmethod
    def chi4_cdf(x: float) -> float:
        if x <= 0.0:
            return 0.0
        return 1.0 - math.exp(-0.5 * x) * (1.0 + 0.5 * x)

    @staticmethod
    def chi1_pdf(x: np.ndarray | float) -> np.ndarray:
        x = np.asarray(x)
        out = np.zeros_like(x, dtype=float)
        mask = x > 0.0
        out[mask] = np.exp(-0.5 * x[mask]) / np.sqrt(2.0 * math.pi * x[mask])
        return out

    def weighted_chi_pdf(self, x: np.ndarray | float, alpha: float) -> np.ndarray:
        x = np.asarray(x)

        if alpha <= 1e-14:
            return self.chi1_pdf(x)

        out = np.zeros_like(x, dtype=float)
        mask = x > 0.0
        xm = x[mask]

        z = xm * abs(1.0 - alpha) / (4.0 * alpha)
        log_pdf = (
            -math.log(2.0 * math.sqrt(alpha))
            - xm * (1.0 + alpha) / (4.0 * alpha)
            + z
            + np.log(special.i0e(z))
        )

        out[mask] = np.exp(log_pdf)
        return out

    def h_bound(self, t: np.ndarray, rho: float) -> np.ndarray:
        return self.chi1_cdf((rho - 1.0) * t / (1.0 - t))

    def g_rho(self, theta: float, rho: float) -> float:
        x = theta ** (-2.0)

        if rho >= self.config.rho_max:
            return self.chi4_cdf(x)

        split = max(0.0, 1.0 + x - rho)

        term1 = self.integrate(
            lambda t: self.h_bound(t, rho) * self.chi1_pdf((x - t) / rho) / rho,
            0.0,
            split,
        )

        term2 = self.integrate(
            lambda t: self.h_bound(t, rho) * self.weighted_chi_pdf(x - t, rho - 1.0),
            split,
            x,
        )

        return term1 + term2

    def sup_g(self, theta: float) -> tuple[float, float]:
        x = theta ** (-2.0)

        grids = [
            np.linspace(1.0, self.config.rho_max, self.config.rho_grid_size),
            np.linspace(1.0, min(self.config.rho_max, 1.0 + x), self.config.rho_grid_size // 2),
        ]

        if 1.0 + x < self.config.rho_max:
            grids.append(
                np.linspace(
                    max(1.0, 1.0 + x - 0.1),
                    min(self.config.rho_max, 1.0 + x + 0.1),
                    self.config.rho_grid_size // 3,
                )
            )

        rho_grid = np.unique(np.concatenate(grids))
        values = np.array([self.g_rho(theta, float(rho)) for rho in rho_grid])

        idx = int(np.argmax(values))
        best_value = float(values[idx])
        best_rho = float(rho_grid[idx])

        left = float(rho_grid[max(0, idx - 3)])
        right = float(rho_grid[min(len(rho_grid) - 1, idx + 3)])

        if right > left:
            result = optimize.minimize_scalar(
                lambda rho: -self.g_rho(theta, float(rho)),
                bounds=(left, right),
                method="bounded",
                options={"xatol": self.config.xatol},
            )

            if -result.fun > best_value:
                best_value = float(-result.fun)
                best_rho = float(result.x)

        rho_max_value = self.chi4_cdf(x)

        if rho_max_value > best_value:
            best_value = rho_max_value
            best_rho = self.config.rho_max

        return best_value, best_rho

    def theta_cb(self, delta: float) -> tuple[float, float, float]:
        def objective(theta: float) -> float:
            return self.sup_g(theta)[0] - delta

        left = 1.0
        right = 2.0

        while objective(right) > 0.0:
            right *= 1.5

        theta = optimize.brentq(objective, left, right, xtol=1e-6, rtol=1e-6)
        value, rho_star = self.sup_g(theta)

        return float(theta), value, rho_star


def theta_vanilla(delta: float, k: int = 3) -> float:
    return math.sqrt(2.0 / math.pi) * delta ** (-1.0 / k)


def main() -> None:
    deltas = [0.1, 0.05, 0.01, 0.001]
    calibrator = CounterbalanceCalibrator()

    print("delta    theta_cb    theta_v    rho_star")
    print("-----------------------------------------")

    for delta in deltas:
        theta_cb, _, rho_star = calibrator.theta_cb(delta)
        theta_v = theta_vanilla(delta)

        print(f"{delta:<8g} {theta_cb:<10.4f} {theta_v:<10.4f} {rho_star:.4f}")


if __name__ == "__main__":
    main()