import numpy as np
import finite_difference as fd
import matplotlib.pyplot as plt



class ExplicitKolmogorov2D():

    def __init__(self, coefficients, boundary, right_hand_side, grid):
        self.coefficients = coefficients
        self.boundary = boundary
        self.grid = grid
        self.right_hand_side = right_hand_side
        self.stability_check = 0
    


    # Function to compute grid, coefficients and right-hand side
    def compute_coefficients(self):
      
        # Define grid
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
        T, X, Y = np.meshgrid(self.T, self.X, self.Y, indexing='ij')
     
        # Compute coefficients
        if np.isscalar(self.coefficients[0]):
            self.a = np.full((self.Nt, self.Nx, self.Ny), self.coefficients[0])
        else:             
            self.a = self.coefficients[0](T, X, Y)
        if np.isscalar(self.coefficients[1]):
            self.b = np.full((self.Nt, self.Nx, self.Ny), self.coefficients[1])
        else: 
            self.b = self.coefficients[1](T, X, Y)
        if np.isscalar(self.coefficients[2]):
            self.c = np.full((self.Nt, self.Nx, self.Ny), self.coefficients[2])
        else:
            self.c = self.coefficients[2](T, X, Y)

        # Check stability condition
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
                self.compute_coefficients()
            else:
                raise ValueError("Error in verifying stability condition. Please check the coefficients and the time step.")

        # Compute right-hand side
        if np.isscalar(self.right_hand_side):
            self.rhs = np.full((self.Nt, self.Nx, self.Ny), self.right_hand_side)
        else:
            self.rhs = self.right_hand_side(T, X, Y)



    # Function to apply initial and boundary conditions
    def compute_boundary_conditions(self):
        self.solution = np.zeros((self.Nt, self.Nx, self.Ny))
        indices = [0, -1]
        for idx, index in enumerate(indices):
            # X boundary conditions
            fx = self.boundary[idx]
            if np.isscalar(fx):
                self.solution[:, index, :] = fx
            else:
                try:
                    self.solution[:, index, :] = fx(self.T[:, None], self.Y[None, :])
                except:
                    raise ValueError("Invalid boundary condition provided.")
        # Y boundary conditions 
        fy_left = self.boundary[2]
        if np.isscalar(fy_left):
            self.solution[:, self.Lx:, 0] = fy_left
        else:
            try:
                self.solution[:, self.Lx:, 0]  = fy_left(self.T[:, None], self.X[None, self.Lx//2:])
            except:
                raise ValueError("Invalid boundary condition provided.")
        fy_right = self.boundary[3]
        if np.isscalar(fy_right):
            self.solution[:, :self.Lx, -1] = fy_right
        else:
            try:
                self.solution[:, :self.Lx, -1]  = fy_right(self.T[:, None], self.X[None, :self.Lx//2])
            except:
                raise ValueError("Invalid boundary condition provided.")
        # Initial condition
        if np.isscalar(self.boundary[4]):
            self.solution[0, :, :] = self.boundary[4]
        else:     
            try:
                self.solution[0, :, :] = self.boundary[4](self.X[:, None], self.Y[None, :])
            except:
                raise ValueError("Invalid initial condition provided.")



    # Method to plot the solution
    def plot_solution(self, pause_time = 0.1):
        X = self.X
        Y = self.Y
        plt.figure()
        for time in range(self.Nt):
            plt.clf()
            plt.contourf(X, Y, self.solution[time, :, :].T, cmap='viridis')
            plt.title(f'Solution at time t={self.T[time]:.2f}')
            plt.colorbar(label='u(t, x, y)')
            plt.pause(pause_time)
        plt.show()



    # Method to solve the PDE
    def solve(self, dx, dt = 0.1):
        self.dx = dx
        self.dt = dt
        self.compute_coefficients()
        self.compute_boundary_conditions()      
        percentages = 0  
        print(f"Computing solution: {percentages}%", end="")
        for n in range(self.Nt - 1):
            for i in range(1, self.Nx - 1):
                jmin = max(1, self.Lx - i) * (n + 1)
                jmax = (self.Ny - 1) - max(0, i - self.Lx) * (n + 1)
                for j in range(jmin, jmax + 1):
                    Lie = i + j - self.Lx
                    self.solution[n + 1, i, j] = self.dt * (self.a[n, i, j] * fd.second_order_difference(self.solution[n, :, Lie], self.dx, i) +
                                        self.b[n, i, j] * fd.central_difference(self.solution[n, :, Lie], self.dx, i) +
                                        self.c[n, i, j] * self.solution[n, i, j] - self.rhs[n, i, j]) + self.solution[n, i, Lie]      
            if int((n + 1) / self.Nt * 100) > percentages:
                percentages = int((n + 1) / self.Nt * 100)
                print(f"\rComputing solution: {percentages}%", end="")
        print("\nSolution computed.")
        return 
                    


if __name__ == "__main__":
    grid = (3, 3, 1)
    boundary = (2, 2, 2, 2, lambda x, y: np.exp(-1 * (x**2 + y**2)))
    coefficients = (0.1, 0, 0)
    right_hand_side = 0
    kolmogorov = ExplicitKolmogorov2D(coefficients, boundary, right_hand_side, grid)
    kolmogorov.solve(0.1, 0.1)
    kolmogorov.plot_solution()
    input("Test completed.")

