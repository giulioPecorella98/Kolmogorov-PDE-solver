import unittest
import numpy as np
from src.kolmogorov2D.solver import ExplicitSolver   

def make_solver(grid, boundary, coeffs, rhs):
    solver = ExplicitSolver(grid, coeffs)
    solver.initialize(0.5, 0.1)
    solver.compute_boundary_conditions(boundary)
    solver.compute_right_hand_side(rhs)
    solver.solve(verbose=False)
    return solver

class TestKolmogorovSolution(unittest.TestCase):

    def test_works(self):
        grid = (5, 5, 3)
        boundary = (0, 0, 0, 0, lambda x, y: np.exp(-1 * (x**2 + y**2)))
        coefficients = (1, 0, 0)
        rhs = 0
        kolmogorov = make_solver(grid, boundary, coefficients, rhs)
        self.assertIsNotNone(kolmogorov.solution)
        self.assertEqual(kolmogorov.solution.shape, (kolmogorov.Nt, kolmogorov.Nx, kolmogorov.Ny))
        self.assertTrue(np.isfinite(kolmogorov.solution).all())

    def test_zero_solution(self):
        grid = (5, 5, 3)
        boundary = (0, 0, 0, 0, 0)
        coefficients = (1, 0, 0)
        rhs = 0
        kolmogorov = make_solver(grid, boundary, coefficients, rhs)
        self.assertAlmostEqual(np.max(kolmogorov.solution), 0)
        self.assertAlmostEqual(np.min(kolmogorov.solution), 0)
        self.assertTrue(np.isfinite(kolmogorov.solution).all())

    def test_maximum_principle(self):
        grid = (5, 5, 3)
        boundary = (lambda t, y: np.exp(-1 * (t**2 + y**2)), 
                    lambda t, y: np.exp(-1 * (t**2 + y**2)), 
                    lambda t, x: np.exp(-1 * (t**2 + x**2)), 
                    lambda t, x: np.exp(-1 * (t**2 + x**2)), 
                    lambda x, y: np.exp(-1 * (x**2 + y**2)))
        coefficients = (1, 0, 0)
        rhs = 0
        kolmogorov = make_solver(grid, boundary, coefficients, rhs)
        self.assertGreaterEqual(kolmogorov.solution.min(), -1e-8)
        self.assertTrue(np.isfinite(kolmogorov.solution).all())

    def test_variable_coefficients(self):
        grid = (3, 3, 1)
        boundary = (0, 0, 0, 0, 0)
        coefficients = (lambda t, x, y: t + x + y,
                        lambda t, x, y: t + x + y, 
                        lambda t, x, y: t + x + y)
        rhs = lambda t, x, y: np.exp(-1 * (t + x + y))
        kolmogorov = make_solver(grid, boundary, coefficients, rhs)
        self.assertIsNotNone(kolmogorov.solution)
        self.assertEqual(kolmogorov.solution.shape, (kolmogorov.Nt, kolmogorov.Nx, kolmogorov.Ny))
        self.assertTrue(np.isfinite(kolmogorov.solution).all())