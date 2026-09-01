from src.kolmogorov2d.solver import ExplicitSolver
from src.kolmogorov2d.visualization import plot_solution
import numpy as np

if __name__ == "__main__":
    grid = (1, 5, 5)
    boundary = (0, 0, 0, 0, lambda x, y: np.exp(-1 * (x**2 + y**2)))
    coefficients = (1, 0, 0)
    right_hand_side = 0
    kolmogorov = ExplicitSolver(grid, coefficients)
    kolmogorov.initialize(0.5, 0.1)
    kolmogorov.compute_boundary_conditions(boundary)
    kolmogorov.compute_right_hand_side(right_hand_side)
    kolmogorov.solve()
    solution = kolmogorov.return_solution()
    T, X, Y = kolmogorov.return_grid()
    plot_solution(T, X, Y, solution)
    input("Explicit completed.")