import unittest

from theta_tuning import (
    PAPER_THETA,
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

    def test_manuscript_rows_match_at_printed_precision(self) -> None:
        expected = {
            0.1: ("0.094291", "1.2391"),
            0.05: ("0.045134", "1.1631"),
            0.01: ("0.008700", "1.0577"),
            0.001: ("0.000861", "1.0126"),
        }
        for delta, theta in PAPER_THETA.items():
            with self.subTest(delta=delta):
                result = matrix_agnostic_bound(theta)
                self.assertEqual(f"{result.value:.6f}", expected[delta][0])
                self.assertEqual(f"{result.rho_at_max:.4f}", expected[delta][1])
                self.assertEqual(result.branch, "1 < rho < 7")

    def test_parameter_domain_is_checked(self) -> None:
        with self.assertRaises(ValueError):
            coupled_bound(theta=1.0, alpha=0.2, beta=0.01)
        with self.assertRaises(ValueError):
            coupled_bound(theta=1.76, alpha=0.2, beta=1.0)


if __name__ == "__main__":
    unittest.main()
