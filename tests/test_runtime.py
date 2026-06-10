import unittest
from src.kolmogorov2d.solver import ExplicitSolver   

def make_solver(grid, boundary, coeffs, rhs):
    solver = ExplicitSolver(grid, coeffs)
    solver.initialize(0.5, 0.1)
    solver.compute_boundary_conditions(boundary)
    solver.compute_right_hand_side(rhs)
    solver.solve(verbose=False)
    return solver

good_grid = (3, 3, 1)
good_boundary = (0, 0, 0, 0, 0)
good_coefficients = (1, 0, 0)
good_rhs = 0

class TestKolmogorovInput(unittest.TestCase):
    
    def test_boundary_first(self):
        good_kolmogorov = ExplicitSolver(good_grid, good_coefficients)
        with self.assertRaises(RuntimeError):
            good_kolmogorov.compute_boundary_conditions(good_boundary)
            
    def test_few_boundary(self):
        good_kolmogorov = ExplicitSolver(good_grid, good_coefficients)
        with self.assertRaises(ValueError):
            good_kolmogorov.initialize(0.5)
            good_kolmogorov.compute_boundary_conditions((0, 0))
    
    def test_bad_boundary(self):
        good_kolmogorov = ExplicitSolver(good_grid, good_coefficients)
        with self.assertRaises(ValueError):
            good_kolmogorov.initialize(0.5)
            good_kolmogorov.compute_boundary_conditions((0, 0, 0, 0, ()))

    def test_rhs_first(self):
        good_kolmogorov = ExplicitSolver(good_grid, good_coefficients)
        with self.assertRaises(RuntimeError):
            good_kolmogorov.compute_right_hand_side(good_rhs)

    def test_bad_rhs(self):
        good_kolmogorov = ExplicitSolver(good_grid, good_coefficients)
        with self.assertRaises(ValueError):
            good_kolmogorov.initialize(0.5)
            good_kolmogorov.compute_right_hand_side(())

    def test_solve_first(self):
        good_kolmogorov = ExplicitSolver(good_grid, good_coefficients)
        with self.assertRaises(RuntimeError):
            good_kolmogorov.solve(verbose=False)

    def test_solve_no_rhs(self):
        good_kolmogorov = ExplicitSolver(good_grid, good_coefficients)
        with self.assertRaises(RuntimeError):
            good_kolmogorov.initialize(0.5)
            good_kolmogorov.compute_boundary_conditions((0, 0, 0, 0, 0))
            good_kolmogorov.solve(verbose=False)

    def test_no_recomputing_rhs(self):
        good_kolmogorov = ExplicitSolver(good_grid, good_coefficients)
        with self.assertRaises(RuntimeError):
            good_kolmogorov.initialize(0.5)
            good_kolmogorov.compute_boundary_conditions((0, 0, 0, 0, 0))
            good_kolmogorov.compute_right_hand_side(0)
            good_kolmogorov.initialize(0.3)
            good_kolmogorov.compute_boundary_conditions((0, 0, 0, 0, 0))
            good_kolmogorov.solve(verbose=False)

    def test_no_recomputing_boundary(self):
        good_kolmogorov = ExplicitSolver(good_grid, good_coefficients)
        with self.assertRaises(RuntimeError):
            good_kolmogorov.initialize(0.5)
            good_kolmogorov.compute_boundary_conditions((0, 0, 0, 0, 0))
            good_kolmogorov.compute_right_hand_side(0)
            good_kolmogorov.initialize(0.3)
            good_kolmogorov.compute_right_hand_side(0)
            good_kolmogorov.solve(verbose=False)
