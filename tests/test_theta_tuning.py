import unittest

from theta_tuning import (
    CalibrationConfig,
    coupled_bound,
    matrix_agnostic_bound,
)


class ThetaTuningTests(unittest.TestCase):
    def test_quadrature_converges_at_manuscript_maximizer(self) -> None:
        theta = 1.76
        alpha = 0.1628
        fraction = 0.1766
        beta = fraction * alpha * min(alpha, 1.0)
        values = [
            coupled_bound(theta, alpha, beta, order)
            for order in (48, 96, 192)
        ]
        self.assertLess(max(values) - min(values), 2e-5)
        self.assertAlmostEqual(values[-1], 0.0451337, delta=2e-5)

    def test_numerical_maximum_matches_manuscript_check(self) -> None:
        config = CalibrationConfig(
            optimization_order=32,
            verification_orders=(64, 96),
            optimizer_seeds=(617,),
            optimizer_tolerance=1e-4,
            optimizer_population=8,
            optimizer_iterations=45,
        )
        result = matrix_agnostic_bound(1.76, config)
        self.assertAlmostEqual(result.value, 0.0451338, delta=3e-4)
        self.assertAlmostEqual(result.rho_at_max, 1.163, delta=0.015)
        self.assertEqual(result.branch, "1 < rho < 7")

    def test_parameter_domain_is_checked(self) -> None:
        with self.assertRaises(ValueError):
            coupled_bound(theta=1.0, alpha=0.2, beta=0.01)
        with self.assertRaises(ValueError):
            coupled_bound(theta=1.76, alpha=0.2, beta=1.0)


if __name__ == "__main__":
    unittest.main()
