"""
Finite-difference solver for two-dimensional Kolmogorov equations

u_t - x u_y = a(t, x, y) u_xx + b(t, x, y) u_x + c(t, x, y) u - f(t, x, y)

on the domain (0,T) x (-X,X) x (-Y,Y). The solver uses an explicit scheme 
adapted  to the geometric structure of the Lie derivative u_t - x u_y

Type Aliases
------------
Function3D
    Scalar value or function of (t, x, y)

Coefficients3D
    Sequence of PDE coefficients a(t, x, y), b(t, x, y), c(t, x, y) 
    represented as scalars or functions of (t, x, y)

BoundaryFunction
    Scalar value or function of (x, y), (t, x) or (t, y)

Boundary2D
    Sequence of boundary conditions and initial condition represented as 
    scalars or functions of (x, y), (t, x) or (t, y)
"""

import numpy as np
from .finite_difference import second_order_difference as sd
from .finite_difference import central_difference as cd
from typing import TypeAlias, Callable

Function3D: TypeAlias = float | Callable[[float, float, float], float]
Coefficients3D: TypeAlias = tuple[Function3D, Function3D, Function3D]
BoundaryFunction: TypeAlias = float | Callable[[float, float], float]
Boundary2D: TypeAlias = tuple[BoundaryFunction, BoundaryFunction, 
                        BoundaryFunction, BoundaryFunction, BoundaryFunction]



class ExplicitSolver:
    """
    Numerical solver for a 2D Kolmogorov PDE on an origin-centered 
    rectangular domain using an explicit finite difference scheme.
    ...

    Attributes
    ----------
    grid : Sequence[float, float, float]
        a sequence of three floats T, X, Y representing 
        the domain (0, T) x (-X, X) x (-Y, Y)
    T : final time of the domain
    X : maximum lenght in the x variable
    Y : maximum lenght in the y variable
    coefficients : Coefficients3D
        a sequence of three scalars or functions representing 
        the coefficients of the PDE
    rhs : Function3D
        a scalar or function representing the right-hand side of the PDE
    solution : np.ndarray
        a 3D array representing the solution of the PDE

    Methods (first call initialize to set the grid, and call solve only 
             after compute_boundary_conditions and compute_right_hand_side)
    -------
    initialize(dx, dt = 0.1)
        Computes the grid, the coefficients and check the stability condition.
        \\If you call this method again (for instance to refine the grid), you
        need to call again compute_boundary_conditions() and 
        compute_right_hand_side()


    compute_boundary_conditions(boundary)
        Computes boundary conditions and initial condition
    compute_right_hand_side(rhs)
        Computes the right-hand side
    solve()
        Solves the PDE using a subRiemannian finite difference method   
    """

    def __init__(self, grid : tuple[float, float, float], 
                 coefficients : Coefficients3D):
        """
        Parameters
        ----------
        grid : Sequence[float, float, float]
            a sequence of three floats T, X, Y representing 
            the domain (0,T) x (-X, X) x (-Y, Y)
        coefficients : Coefficients3D
            a tuple of three scalars or functions representing the coefficients

        Raises
        ------
        ValueError
            -If the input dimensions are invalid\\
            -If one of the coefficients is not a scalar or a function 
             that can be evaluated\\
            -If one of the grid values is not a float nor positive
        """

        if (len(grid) != 3) or (len(coefficients) != 3):
            raise ValueError("Invalid input dimensions.")
        for i in range(3):
            if not (np.isscalar(coefficients[i]) or callable(coefficients[i])):
                raise ValueError("Invalid coefficient provided. Coefficient "\
                                 "must be a scalar or a function.")
        for i in range(3):
            if not (isinstance(grid[i], (int, float))):
                raise ValueError("Invalid grid value provided. Grid value " \
                                 "must be float or int.")
            if grid[i] < 0:
                raise ValueError("Invalid grid value, lenght must be positive")
        self.grid = grid
        self.coefficients = coefficients
        self.stability_check = 0
        self.T = None
        self.X = None
        self.Y = None
        self.a = None
        self.b = None
        self.c = None
        self.Nt = None
        self.Nx = None
        self.Ny = None
        self.solution = None
        self.rhs = None
        self.computed_solution = False


    def initialize(self, dx: float, dt: float = 0.1) -> None:
        """
        Computes the grid, the coefficients and checks the stability condition
        
        Parameters
        ----------
        dx : float
            The spatial step size
        dt : float, optional
            The time step size (default is 0.1)

        Raises
        ------
        TypeError
            -If the spatial and time step sizes are not numeric
        ValueError
            -If the spatial and time step sizes are not positive\\
            -If the second order coefficient a(t, x , y) is not 
             strictly positive\\
            -If the time step is not small enough to satisfy the stability 
             conditions
        """
        if not isinstance(dx,(int, float)) or not isinstance(dt,(int, float)):
            raise TypeError("dx and dt must be numeric.")
        if dx <= 0 or dx > self.grid[1] or dt <= 0 or dt > self.grid[0]:
            raise ValueError("dx and dt must be valid for the grid.")
        self.dx = dx
        self.dt = dt
        t = self.grid[0]
        self.T = np.arange(0, t + self.dt/2, self.dt)
        self.Nt = len(self.T)
        x = self.grid[1]
        self.Lx = int(x / self.dx)
        self.X, self.dx = np.linspace(-x, x, 2 * self.Lx + 1, 
                                      endpoint=True, retstep=True)
        self.Nx = len(self.X)
        # required by this particular finited difference method
        self.dy = self.dx * self.dt     
        y = self.grid[2]
        self.Y = np.arange(-y, y + self.dy/2, self.dy)
        self.Ny = len(self.Y)
        self.T, self.X, self.Y = np.meshgrid(self.T, self.X, 
                                             self.Y, indexing='ij')

        arrays = []
        for coeff in self.coefficients:
            if np.isscalar(coeff):
                arrays.append(np.full((self.Nt, self.Nx, self.Ny), coeff))
            else:
                arrays.append(coeff(self.T,self.X,self.Y))
        self.a, self.b, self.c = arrays

        a_max = np.max(np.abs(self.a))
        if a_max <= 0:
            raise ValueError("The second order coefficient a(t, x, y) " \
                             "must be strictly positive.")
        b_max = np.max(np.abs(self.b))
        c_max = np.max(np.abs(self.c))

        conditions = [self.dx**2 / (2 * a_max)]
        if b_max > 0:
            conditions.append(self.dx / b_max)
        if c_max > 0:
            conditions.append(2 / c_max)
        stability_condition = min(conditions)
        # Necessary recursion when dealing with non-constant coefficients, as 
        # the stability condition may change after adjusting the time step 
        # (hence the grid)
        if (self.dt >= stability_condition):
            if (self.stability_check < 5):
                self.dt = stability_condition * 0.9
                self.stability_check += 1
                self.initialize(self.dx, self.dt)    
            else:
                raise ValueError("Error in verifying stability condition. " \
                     "Please check the coefficients and verify the time step.")


    def compute_boundary_conditions(self, boundary: Boundary2D) -> None:
        """
        Computes boundary conditions and initial condition

        Parameters        
        ----------
        boundary : Boundary2D
            a tuple of five 2D functions or scalars:
            - fx_left is the left boundary condition (x = -X)
            - fx_right is the right boundary condition (x = X)
            - fy_left is the bottom boundary condition (y = -Y)
            - fy_right is the top boundary condition (y = Y)
            - initial is the initial condition (t = 0)

        Raises
        ------
        RuntimeError
            -If initialize() have not been called before
        ValueError
            -If the input dimensions are invalid
            -If one of the boundary conditions is not a scalar or 
             a function that can be evaluated
        """
        
        if self.X is None:
            raise RuntimeError("Initialize must be called before " \
                               "computing boundary conditions.")
        if len(boundary) != 5:
            raise ValueError("Invalid input dimensions.")
        for i in range(5):
            if not (np.isscalar(boundary[i]) or callable(boundary[i])):
                raise ValueError("Invalid boundary condition provided. Each " \
                          "boundary condition must be a scalar or a function.")
        self.solution = np.zeros((self.Nt, self.Nx, self.Ny))
        indices = [0, -1]
        for idx, index in enumerate(indices):
            fx = boundary[idx]
            if np.isscalar(fx):
                self.solution[:, index, :] = fx
            else:
                t_boundary = self.T[:, index, :]
                y_boundary = self.Y[:, index, :]
                self.solution[:, index, :] = fx(t_boundary, y_boundary)
        fy_left = boundary[2]
        if np.isscalar(fy_left):
            self.solution[:, self.Lx:, 0] = fy_left
        else:
            t_boundary = self.T[:, self.Lx:, 0]
            x_boundary = self.X[:, self.Lx:, 0]
            self.solution[:, self.Lx:, 0] = fy_left(t_boundary, x_boundary)
        fy_right = boundary[3]
        if np.isscalar(fy_right):
            self.solution[:, :self.Lx, -1] = fy_right
        else:
            t_boundary = self.T[:, :self.Lx, -1]
            x_boundary = self.X[:, :self.Lx, -1]
            self.solution[:, :self.Lx, -1] = fy_right(t_boundary, x_boundary)
        if np.isscalar(boundary[4]):
            self.solution[0, :, :] = boundary[4]
        else:
            x_initial = self.X[0, :, :]
            y_initial = self.Y[0, :, :]
            self.solution[0, :, :] = boundary[4](x_initial, y_initial)


    def compute_right_hand_side(self, right_hand_side : Function3D) -> None:
        """
        Computes the right-hand side of the PDE 
        
        Parameters
        ----------
        right_hand_side : Function3D
            a scalar or function representing the right-hand side of the PDE 
        
        Raises
        ------
        RuntimeError
            -If initialize() have not been called before
        ValueError
            -If the input is not a scalar or a function that can be evaluated
        """

        if self.X is None:
            raise RuntimeError("Initialize must be called before " \
                               "computing boundary conditions.")
        if np.isscalar(right_hand_side):
            self.rhs = np.full((self.Nt, self.Nx, self.Ny), right_hand_side)
        elif callable(right_hand_side):
            self.rhs = right_hand_side(self.T, self.X, self.Y)
        else:
            raise ValueError("Right-hand side must be a scalar or a function.")
            

    def solve(self) -> None:
        """ 
        Solves the PDE using a subRiemannian finite difference method, relying
        on the first order approximation of the Lie derivative x p_y - p_t
        
        Raises
        ------
        RuntimeError
            -If the boundary conditions or the right-hand side have not been 
             computed before calling this method.
            -If the shapes of the solution and the right-hand side do not match
        """

        if self.solution is None:
            raise RuntimeError("Boundary conditions must be computed before.")
        if self.rhs is None:
            raise RuntimeError("Right-hand side must be computed before.")
        if self.solution.shape != (self.Nt, self.Nx, self.Ny):
            raise RuntimeError("Please compute again the boundary conditions.")
        if self.rhs.shape != (self.Nt, self.Nx, self.Ny):
            raise RuntimeError("Please compute again the right-hand side.")
        
        percentages = 0  
        print(f"Computing solution: {percentages}%", end="")
        for n in range(self.Nt - 1):
            for i in range(1, self.Nx - 1):
                jmin = max(0, self.Lx - i) * (n + 1)
                jmax = (self.Ny - 1) - max(0, i - self.Lx) * (n + 1)
                for j in range(jmin + 1, jmax + 1):
                    # The Lie derivative is approximated by using the following
                    #  index shift, which is linked to the characteristic flow
                    Lie = i + j - self.Lx
                    diffusion = (self.a[n, i, j] * 
                                 sd(self.solution[n, :, Lie], self.dx, i))
                    drift = (self.b[n, i, j] 
                             * cd(self.solution[n, :, Lie], self.dx, i))
                    reaction = (self.c[n, i, j] * self.solution[n, i, j])
                    self.solution[n + 1, i, j] = (self.solution[n, i, Lie] +
                                                 self.dt * (diffusion + drift +
                                                 reaction - self.rhs[n, i, j]))
            if int((n + 1) / self.Nt * 100) > percentages:
                percentages = int((n + 1) / self.Nt * 100)
                print(f"\rComputing solution: {percentages}%", end="")
        print("\r" + " " * 40, end="") 
        print("\rSolution computed.")        
        self.computed_solution = True
        return 