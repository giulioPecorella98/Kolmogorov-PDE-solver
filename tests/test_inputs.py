import unittest
import numpy as np
from src.kolmogorov2D.solver import ExplicitSolver   

def make_solver(grid, boundary, coeffs, rhs):
    solver = ExplicitSolver(grid, coeffs)
    solver.initialize(0.5, 0.1)
    solver.compute_boundary_conditions(boundary)
    solver.compute_right_hand_side(rhs)
    solver.solve()
    return solver

good_grid = (3, 3, 1)
good_boundary = (0, 0, 0, 0, 0)
good_coefficients = (1, 0, 0)
good_rhs = 0
good_kolmogorov = ExplicitSolver(good_grid, good_coefficients)

class TestKolmogorovInput(unittest.TestCase):

    def test_wrong_grid_size(self):
        grid = (5, 5, 3, 4)
        boundary = (0, 0, 0, 0, lambda x, y: np.exp(-1 * (x**2 + y**2)))
        coefficients = (1, 0, 0)
        rhs = 0
        with self.assertRaises(ValueError):
            make_solver(grid, boundary, coefficients, rhs)

    def test_wrong_grid_input(self):
        grid = (5, 5, -3)
        boundary = (0, 0, 0, 0, lambda x, y: np.exp(-1 * (x**2 + y**2)))
        coefficients = (1, 0, 0)
        rhs = 0
        with self.assertRaises(ValueError):
            make_solver(grid, boundary, coefficients, rhs)

    def test_wrong_coefficients_size(self):
        grid = (5, 5, 3)
        boundary = (0, 0, 0, 0, lambda x, y: np.exp(-1 * (x**2 + y**2)))
        coefficients = (1, 0, 0, 0)
        rhs = 0
        with self.assertRaises(ValueError):
            make_solver(grid, boundary, coefficients, rhs)

    def test_negative_a_coefficient(self):
        grid = (5, 5, 3)
        boundary = (0, 0, 0, 0, lambda x, y: np.exp(-1 * (x**2 + y**2)))
        coefficients = (-1, 0, 0, 0)
        rhs = 0
        with self.assertRaises(ValueError):
            make_solver(grid, boundary, coefficients, rhs)

    def test_wrong_coefficients_input(self):
        grid = (5, 5, -3)
        boundary = (0, 0, 0, 0, lambda x, y: np.exp(-1 * (x**2 + y**2)))
        coefficients = (1, 0, [0, 0, 0])
        rhs = 0
        with self.assertRaises(ValueError):
            make_solver(grid, boundary, coefficients, rhs)

    def test_wrong_type_dx(self):
        with self.assertRaises(TypeError):
            good_kolmogorov.initialize('a')
    
    def test_negative_value_dx(self):
        with self.assertRaises(ValueError):
            good_kolmogorov.initialize(-4)

    def test_large_value_dx(self):
        with self.assertRaises(ValueError):
            good_kolmogorov.initialize(40)

    def test_wrong_type_dt(self):
        with self.assertRaises(TypeError):
            good_kolmogorov.initialize(0.1, 'a')
    
    def test_negative_value_dt(self):
        with self.assertRaises(ValueError):
            good_kolmogorov.initialize(0.1, -4)

    def test_large_value_dt(self):
        with self.assertRaises(ValueError):
            good_kolmogorov.initialize(0.1, 40) 