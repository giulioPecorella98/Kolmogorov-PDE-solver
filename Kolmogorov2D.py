"""
Finite-difference solver for two-dimensional Kolmogorov equations

u_t - x u_y = a(t,x,y) u_xx + b(t,x,y) u_x + c(t,x,y) u - f(t,x,y)

on the domain (-X,X) x (-Y,Y) x (0,T).  The solver uses an explicit finite-difference
scheme adapted to the geometric structure of the Lie derivative u_t - x u_y


Type Aliases
------------
Function3D
    Scalar value or function of (t, x, y).

Coefficients3D
    Sequence of PDE coefficients represented as scalars or functions
    of (t, x, y).

Function2D
    Scalar value or function of (x, y).

Boundary2D
    Sequence of boundary or initial conditions represented as scalars
    or functions of (x, y), (x,t) or (t,y).
"""
import numpy as np
import finite_difference as fd
import matplotlib.pyplot as plt
from typing import TypeAlias, Callable

Function3D: TypeAlias = float | Callable[[float, float, float], float]
Coefficients3D: TypeAlias = tuple[Function3D, Function3D, Function3D]
BoundaryFunction: TypeAlias = float | Callable[..., float]
Boundary2D: TypeAlias = tuple[BoundaryFunction, BoundaryFunction, BoundaryFunction, BoundaryFunction, BoundaryFunction]



class ExplicitSolver:
    """
    Numerical solver for a two-dimensional Kolmogorov PDE on an origin-centered 
    rectangular domain using an explicit finite difference scheme.

    ...

    Attributes
    ----------
    grid : Sequence[float, float, float]
        a sequence of three floats X, Y, T representing the domain (-X, X) x (-Y, Y) x (0, T)
    coefficients : Coefficients3D
        a sequence of three scalars or functions representing the coefficients of the PDE
    rhs : Function3D
        a scalar or function representing the right-hand side of the PDE
    solution : np.ndarray
        a 3D array representing the solution of the PDE at each time step and spatial point
        
    Methods (to be called in order)
    -------
    compute_coefficients(dx, dt = 0.1)
        Computes the grid, the coefficients and check the stability condition
    compute_boundary_conditions(boundary)
        Computes the initial and boundary conditions
    compute_right_hand_side(rhs)
        Computes the right-hand side
    solve()
        Solves the PDE using a subRiemannian finite difference method
    plot_solution(pause_time = 0.1)
        Plots the solution at each time step
        
    """

    def __init__(self, grid : tuple[float, float, float], coefficients : Coefficients3D):
        """
        Parameters
        ----------
        grid : Sequence[float, float, float]
            a sequence of three floats X, Y, T representing the domain (-X, X) x (-Y, Y) x (0, T)
        coefficients : Coefficients3D
            a tuple of three scalars or functions representing the coefficients of the PDE

        Raises
        ------
        ValueError
            -If the input dimensions are invalid.
            -If one of the coefficients is not a scalar or a function that can be evaluated.
            -If one of the grid values is not a float.
        """

        if (len(grid) != 3) or (len(coefficients) != 3):
            raise ValueError("Invalid input dimensions.")
        for i in range(3):
            if not (np.isscalar(coefficients[i]) or callable(coefficients[i])):
                raise ValueError("Invalid coefficient provided. Each coefficient "\
                                 "must be a scalar or a function.")
        for i in range(3):
            if not (isinstance(grid[i], (int, float))):
                raise ValueError("Invalid grid value provided. Each grid value must be a float.")
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


    def compute_coefficients(self, dx: float, dt: float = 0.1) -> None:
        """
        Computes the grid, the coefficients and checks the stability condition.
        
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
            -If the spatial and time step sizes are not positive
            -If the second order coefficient a(x,y,t) is not strictly positive
            -If the time step is not small enough to satisfy the stability conditions
        """
        if not isinstance(dx, (int, float)) or not isinstance(dt, (int, float)):
            raise TypeError("dx and dt must be numeric.")
        if dx <= 0 or dt <= 0:
            raise ValueError("dx and dt must be positive.")
        self.dx = dx
        self.dt = dt
        x = self.grid[0]
        self.Lx = int(x / self.dx)
        self.X, self.dx = np.linspace(-x, x, 2 * self.Lx + 1, endpoint=True, retstep=True)
        self.Nx = len(self.X)
        self.dy = self.dx * self.dt     # required by this particular finited difference method
        y = self.grid[1]
        self.Y = np.arange(-y, y + self.dy/2, self.dy)
        self.Ny = len(self.Y)
        t = self.grid[2]
        self.T = np.arange(0, t + self.dt/2, self.dt)
        self.Nt = len(self.T)
        self.T, self.X, self.Y = np.meshgrid(self.T, self.X, self.Y, indexing='ij')

        arrays = []
        for coeff in self.coefficients:
            if np.isscalar(coeff):
                arrays.append(np.full((self.Nt, self.Nx, self.Ny), coeff))
            else:
                arrays.append(coeff(self.T,self.X,self.Y))
        self.a, self.b, self.c = arrays

        a_max = np.max(np.abs(self.a))
        if a_max <= 0:
            raise ValueError("The second order coefficient a must be strictly positive.")
        b_max = np.max(np.abs(self.b))
        c_max = np.max(np.abs(self.c))

        conditions = [self.dx**2 / (2 * a_max)]
        if b_max > 0:
            conditions.append(self.dx / b_max)
        if c_max > 0:
            conditions.append(2 / c_max)
        stability_condition = min(conditions)
        """
        necessary recursion when dealing with non-constant coefficients, since 
        the stability condition may change after adjusting the time step (hence the grid)
        """
        if (self.dt >= stability_condition):
            if (self.stability_check < 5):
                self.dt = stability_condition * 0.9
                self.stability_check += 1
                self.compute_coefficients(self.dx, self.dt)    
            else:
                raise ValueError("Error in verifying stability condition. Please check the coefficients and verify the time step.")





    def compute_boundary_conditions(self, boundary: Boundary2D) -> None:
        """
        Computes the initial and boundary conditions.

        Parameters        
        ----------
        boundary : Boundary2D
            a tuple of five 2D functions or scalars:
            - fx_left is the left boundary condition (x = -X)
            - fx_right is the function or scalar for the right boundary condition (x = X)
            - fy_left is the function or scalar for the bottom boundary condition (y = -Y)
            - fy_right is the function or scalar for the top boundary condition (y = Y)
            - initial is the function or scalar for the initial condition (t = 0)

        Raises
        ------
        RuntimeError
            -If the coefficients have not been computed before calling this method
        ValueError
            -If the input dimensions are invalid
            -If one of the boundary conditions is not a scalar or a function that can be evaluated
        """
        
        if self.X is None:
            raise RuntimeError("Coefficients must be computed before computing boundary conditions.")
        if len(boundary) != 5:
            raise ValueError("Invalid input dimensions.")
        for i in range(5):
            if not (np.isscalar(boundary[i]) or callable(boundary[i])):
                raise ValueError("Invalid boundary condition provided. Each boundary condition " \
                      "must be a scalar or a function.")
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
        Computes the right-hand side of the PDE. 
        
        Parameters
        ----------
        right_hand_side : Function3D
            a scalar or function representing the right-hand side of the PDE 
        
        Raises
        ------
        RuntimeError
            -If the coefficients have not been computed before calling this method
        ValueError
            -If the input is invalid (not a scalar or a function that can be evaluated)
        """

        if self.X is None:
            raise RuntimeError("Coefficients must be computed before computing the right-hand side.")
        if np.isscalar(right_hand_side):
            self.rhs = np.full((self.Nt, self.Nx, self.Ny), right_hand_side)
        elif callable(right_hand_side):
            self.rhs = right_hand_side(self.T, self.X, self.Y)
        else:
            raise ValueError("Invalid right-hand side provided. It must be a scalar or a function.")
            





    def solve(self) -> None:
        """ 
        Solves the PDE using a subRiemannian finite difference method. It relies
        on the first order approximation of the Lie derivative x p_y - p_t
        
        Raises
        ------
        RuntimeError
            -If the coefficients, the boundary conditions or the right-hand side have not been computed before calling this method.
            -If the shapes of the solution and the right-hand side do not match
        """

        if self.X is None:
            raise RuntimeError("Coefficients must be computed before solving the PDE.")
        if self.solution is None:
            raise RuntimeError("Boundary conditions must be computed before solving the PDE.")
        if self.rhs is None:
            raise RuntimeError("Right-hand side must be computed before solving the PDE.")
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
                    # The Lie derivative is approximated by using the following index shift, which is linked to the characteristic flow of the PDE:
                    Lie = i + j - self.Lx
                    self.solution[n + 1, i, j] = self.dt * (self.a[n, i, j] * fd.second_order_difference(self.solution[n, :, Lie], self.dx, i) +
                                        self.b[n, i, j] * fd.central_difference(self.solution[n, :, Lie], self.dx, i) +
                                        self.c[n, i, j] * self.solution[n, i, j] - self.rhs[n, i, j]) + self.solution[n, i, Lie]      
            if int((n + 1) / self.Nt * 100) > percentages:
                percentages = int((n + 1) / self.Nt * 100)
                print(f"\rComputing solution: {percentages}%", end="")
        print("\r" + " " * 40, end="")  # pulisce la riga
        print("\rSolution computed.")        
        self.computed_solution = True
        return 



    def plot_solution(self, pause_time: float = 0.1) -> None:
        """
        Plots the solution at each time step. 

        Parameters
        ----------
        pause_time : float, optional
            The time to pause between each plot (default is 0.1 seconds)
        
        Raises
        ------
        RuntimeError
            If the solution has not been computed before calling this method.
        """
        if not self.computed_solution:
            raise RuntimeError("Please compute the solution before plotting.")
        X = self.X[0, :, :]
        Y = self.Y[0, :, :]
        plt.figure()
        for time in range(self.Nt):
            plt.clf()
            plt.contourf(X, Y, self.solution[time, :, :], cmap='viridis')
            plt.title(f'Solution at time t={self.T[time, 0, 0]:.2f}')
            plt.colorbar(label=f'u(x, y, {self.T[time, 0, 0]:.2f})')
            plt.pause(pause_time)
        plt.show()





if __name__ == "__main__":
    grid = (5, 5, 3)
    boundary = (0, 0, 0, 0, lambda x, y: np.exp(-1 * (x**2 + y**2)))
    coefficients = (1, 0, 0)
    right_hand_side = 0
    kolmogorov = ExplicitSolver(grid, coefficients)
    kolmogorov.compute_coefficients(0.5, 0.01)
    kolmogorov.compute_boundary_conditions(boundary)
    kolmogorov.compute_right_hand_side(right_hand_side)
    kolmogorov.solve()
    kolmogorov.plot_solution()
    input("Test completed.")

