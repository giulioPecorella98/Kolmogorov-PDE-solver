# Kolmogorov 2D PDE Solver

A Python library for solving **degenerate 2D Kolmogorov-type PDEs** using 
finite difference schemes.

## Overview

This package provides numerical solvers for linear parabolic PDEs of the form:

<p align="center">
  <img src="https://raw.githubusercontent.com/giulioPecorella98/Kolmogorov-PDE-solver/main/docs/operator.png" alt="Kolmogorov PDE">
</p>

on the domain **(0,T) × (-X,X) × (-Y,Y)**.

The numerical scheme implements a specialized finite difference method adapted 
to the geometric structure of the Lie derivative, providing accurate 
approximations of the derivative:

<p align="center">
  <img src="https://raw.githubusercontent.com/giulioPecorella98/Kolmogorov-PDE-solver/main/docs/directional_derivative.png" alt="Lie Derivative">
</p>

using the approximation:

<p align="center">
  <img src="https://raw.githubusercontent.com/giulioPecorella98/Kolmogorov-PDE-solver/main/docs/Lie_approximation.png" alt="Lie Approximation Scheme">
</p>

For more details about the approximation method and the boundary conditions 
please refer to [https://link.springer.com/article/10.1007/BF02575835] .

## Features

- Explicit method for 2D Kolmogorov equations
- Implicit method (to be implemented) for 2D Kolmogorov equations
- NumPy-based for efficient numerical computation
- Pure Python implementation with minimal dependencies

## Installation

### From PyPI (soon available)

```bash
pip install kolmogorov2d
```

### From source

```bash
git clone https://github.com/giulioPecorella98/Kolmogorov-PDE-solver.git
cd Kolmogorov-PDE-solver
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

**Requirements:**
- Python ≥ 3.11
- NumPy
- Matplotlib (for data visualization only)

## Quick example

```python
from kolmogorov2d.solver import ExplicitSolver
import numpy as np

# Define domain: (0, T) × (-X, X) × (-Y, Y)
domain = (1.0, 1.0, 1.0)  # T=1, X=1, Y=1

# Define PDE coefficients a(t,x,y), b(t,x,y), c(t,x,y)
coefficients = (1, 0, 0)  #constant coefficients

# Create solver instance
solver = ExplicitSolver(domain, coefficients)

# Initialize grid
solver.initialize(0.5, 0.1)   #dx=0.5, dt=0.1 

# Define and compute boundary and initial conditions
boundary = (0.0, 0.0, 0.0, 0.0, lambda x, y: np.exp(-(x**2 + y**2)))
solver.compute_boundary_conditions(boundary)

# Define and compute right-hand side
rhs = 0 
solver.compute_right_hand_side(rhs)

# Solve the PDE
solver.solve()
solution = solver.return_solution()
```

## Package Structure

```
kolmogorov2d/
├── solver.py              # Solver class
├── finite_difference.py   # Finite difference operators
├── visualization.py       # Plotting utilities
└── __init__.py            # Package initialization
```

## Module Documentation

### `solver.py`

Contains the main `ExplicitSolver` class for solving the equation with an 
explicit scheme:

### `finite_difference.py`

Implements finite difference approximations for derivatives

### `visualization.py`

Utilities for visualizing numerical solutions using Matplotlib.

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) 
file for details.

## Contributing

Contributions are welcome! Please feel free to:
- Report bugs through GitHub Issues
- Submit pull requests with improvements
- Suggest features or enhancements

## Author

**Giulio Pecorella**  
giuliopecorella98@gmail.com

## Citation

If you use this solver in your research, please cite:

```bibtex
@software{pecorella_2026_kolmogorov,
  title={Kolmogorov 2D PDE Solver},
  author={Pecorella, Giulio},
  year={2026},
  url={https://github.com/giulioPecorella98/Kolmogorov-PDE-solver}
}
```