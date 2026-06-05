# TODO continuare a sistemare la documentazione, ed aggiungere i check sulla lunghezza delle tuple

"""
Finite-difference solver for two-dimensional Kolmogorov equations.

Type Aliases
------------
Function3D
    Scalar value or function of (t, x, y).

Coefficients3D
    Sequence of PDE coefficients represented as scalars or functions
    of (t, x, y).

Function2D
    Scalar value or function of (x, y).

Coefficients2D
    Sequence of boundary or initial conditions represented as scalars
    or functions of (x, y).
"""
import numpy as np
import finite_difference as fd
import matplotlib.pyplot as plt
from typing import TypeAlias

Function3D: TypeAlias = float | function[[float, float, float], float]
Coefficients3D: TypeAlias = tuple[Function3D]
Function2D: TypeAlias = float | function[[float, float], float]
Coefficients2D: TypeAlias = tuple[Function2D]



class ExplicitKolmogorov2D():
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
        
    Methods
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
            a sequence of three scalars or functions representing the coefficients of the PDE
        """

        self.grid = grid
        self.coefficients = coefficients
        self.stability_check = 0
    


    def compute_coefficients(self, dx: float, dt: float = 0.1):
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
        ValueError
            If the stability condition is not satisfied after 5 attempts to adjust the time step.
        """

        self.dx = dx
        self.dt = dt
        x = self.grid[0]
        self.Lx = int(x / self.dx)
        self.X, self.dx = np.linspace(-x, x, 2 * self.Lx + 1, endpoint=True, retstep=True)
        self.Nx = len(self.X)
        self.dy = self.dx * self.dt
        y = self.grid[1]
        self.Y = np.arange(-y, y + self.dy/2, self.dy)
        self.Ny = len(self.Y)
        t = self.grid[2]
        self.T = np.arange(0, t + self.dt/2, self.dt)
        self.Nt = len(self.T)
        self.T, self.X, self.Y = np.meshgrid(self.T, self.X, self.Y, indexing='ij')

        if np.isscalar(self.coefficients[0]):
            self.a = np.full((self.Nt, self.Nx, self.Ny), self.coefficients[0])
        else:             
            self.a = self.coefficients[0](self.T, self.X, self.Y)
        if np.isscalar(self.coefficients[1]):
            self.b = np.full((self.Nt, self.Nx, self.Ny), self.coefficients[1])
        else: 
            self.b = self.coefficients[1](self.T, self.X, self.Y)
        if np.isscalar(self.coefficients[2]):
            self.c = np.full((self.Nt, self.Nx, self.Ny), self.coefficients[2])
        else:
            self.c = self.coefficients[2](self.T, self.X, self.Y)

        a_max = np.max(np.abs(self.a))
        b_max = np.max(np.abs(self.b))
        c_max = np.max(np.abs(self.c))
        if b_max == 0:
            if c_max == 0:
                stability_condition = self.dx ** 2 / (2 * a_max)
            else:
                stability_condition = min(self.dx ** 2 / (2 * a_max), 2 / c_max)
        else:
            if c_max == 0:
                stability_condition = min(self.dx ** 2 / (2 * a_max), self.dx / b_max)
            else:
                stability_condition = min(self.dx ** 2 / (2 * a_max), self.dx / b_max, 2 / c_max)
        if (self.dt >= stability_condition):
            if (self.stability_check < 5):
                self.dt = stability_condition * 0.9
                self.stability_check += 1
                self.compute_coefficients(self.dx, self.dt)
            else:
                raise ValueError("Error in verifying stability condition. Please check the coefficients and the time step.")



    def compute_boundary_conditions(self, boundary: Coefficients2D):
        """
        Computes the initial and boundary conditions.

        Parameters        
        ----------
        boundary : Coefficients2D
            a sequence of five 2D functions or scalars:
            - fx_left is the left boundary condition (x = -X)
            - fx_right is the function or scalar for the right boundary condition (x = X)
            - fy_left is the function or scalar for the bottom boundary condition (y = -Y)
            - fy_right is the function or scalar for the top boundary condition (y = Y)
            - initial is the function or scalar for the initial condition (t = 0)
        
        Raises
        ------
        ValueError
            If any of the boundary conditions provided is invalid 
            (not a scalar or a function that can be evaluated).
        """

        self.solution = np.zeros((self.Nt, self.Nx, self.Ny))
        indices = [0, -1]
        for idx, index in enumerate(indices):
            fx = boundary[idx]
            if np.isscalar(fx):
                self.solution[:, index, :] = fx
            else:
                try:
                    t_boundary = self.T[:, index, :]
                    y_boundary = self.Y[:, index, :]
                    self.solution[:, index, :] = fx(t_boundary, y_boundary)
                except Exception as exc:
                    raise ValueError("Invalid boundary condition provided.") from exc
        fy_left = boundary[2]
        if np.isscalar(fy_left):
            self.solution[:, self.Lx:, 0] = fy_left
        else:
            try:
                t_boundary = self.T[:, self.Lx:, 0]
                x_boundary = self.X[:, self.Lx:, 0]
                self.solution[:, self.Lx:, 0] = fy_left(t_boundary, x_boundary)
            except Exception as exc:
                raise ValueError("Invalid boundary condition provided.") from exc
        fy_right = boundary[3]
        if np.isscalar(fy_right):
            self.solution[:, :self.Lx, -1] = fy_right
        else:
            try:
                t_boundary = self.T[:, :self.Lx, -1]
                x_boundary = self.X[:, :self.Lx, -1]
                self.solution[:, :self.Lx, -1] = fy_right(t_boundary, x_boundary)
            except Exception as exc:
                raise ValueError("Invalid boundary condition provided.") from exc
        if np.isscalar(boundary[4]):
            self.solution[0, :, :] = boundary[4]
        else:
            try:
                x_initial = self.X[0, :, :]
                y_initial = self.Y[0, :, :]
                self.solution[0, :, :] = boundary[4](x_initial, y_initial)
            except Exception as exc:
                raise ValueError("Invalid initial condition provided.") from exc



    def compute_right_hand_side(self, right_hand_side : Function3D):
        """
        Computes the right-hand side of the PDE.
        
        Parameters
        ----------
        right_hand_side : Function3D
            a scalar or function representing the right-hand side of the PDE
            
        Raises
        ------
        ValueError
            If the right-hand side is not a scalar or a function that can be evaluated.
        """

        if np.isscalar(right_hand_side):
            self.rhs = np.full((self.Nt, self.Nx, self.Ny), right_hand_side)
        else:
            try:
                self.rhs = right_hand_side(self.T, self.X, self.Y)
            except Exception as exc:
                raise ValueError("Invalid right-hand side provided.") from exc


    # Method to solve the PDE
    def solve(self):
        """ Solves the PDE using a subRiemannian finite difference method."""

        percentages = 0  
        print(f"Computing solution: {percentages}%", end="")
        for n in range(self.Nt - 1):
            for i in range(1, self.Nx - 1):
                jmin = max(0, self.Lx - i) * (n + 1)
                jmax = (self.Ny - 1) - max(0, i - self.Lx) * (n + 1)
                for j in range(jmin + 1, jmax + 1):
                    Lie = i + j - self.Lx
                    self.solution[n + 1, i, j] = self.dt * (self.a[n, i, j] * fd.second_order_difference(self.solution[n, :, Lie], self.dx, i) +
                                        self.b[n, i, j] * fd.central_difference(self.solution[n, :, Lie], self.dx, i) +
                                        self.c[n, i, j] * self.solution[n, i, j] - self.rhs[n, i, j]) + self.solution[n, i, Lie]      
            if int((n + 1) / self.Nt * 100) > percentages:
                percentages = int((n + 1) / self.Nt * 100)
                print(f"\rComputing solution: {percentages}%", end="")
        print("\nSolution computed.")
        return 



    def plot_solution(self, pause_time: float = 0.1):
        """
        Plots the solution at each time step.

        Parameters
        ----------
        pause_time : float, optional
            The time to pause between each plot (default is 0.1 seconds)
        
        """
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
    kolmogorov = ExplicitKolmogorov2D(grid, coefficients)
    kolmogorov.compute_coefficients(0.5, 0.1)
    kolmogorov.compute_boundary_conditions(boundary)
    kolmogorov.compute_right_hand_side(right_hand_side)
    kolmogorov.solve()
    kolmogorov.plot_solution()
    input("Test completed.")

