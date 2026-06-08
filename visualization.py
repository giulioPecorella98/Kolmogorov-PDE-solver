import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from matplotlib.figure import Figure

def plot_solution(X: NDArray[np.float64], Y: NDArray[np.float64],
                  T: NDArray[np.float64], solution: NDArray[np.float64], 
                  pause_time: float = 0.1, show: bool = True) -> Figure:        
    """
    Plots the solution at each time step.

    Parameters
    ----------
    X, Y, T : ndarray
        Meshgrid arrays returned by the solver
    solution: ndarray
        Array of shape (Nt, Nx, Ny) containing the numerical solution
    pause_time : float, optional
        The time to pause between each plot (default is 0.1 seconds)
    show : bool, optional
        If True, display the animation

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the animation
    
    Raises
    ------
    ValueError
        -If solution and grid do not match shape, or do not have right shape
    """

    if X.ndim != 3:
        raise ValueError("meshgrid arrays must be 3D.")
    if not (solution.shape == T.shape == X.shape == Y.shape):
        raise ValueError("X, Y, T and solution must have the same shape.")

    T = T[:, 0, 0]
    X = X[0, :, :]
    Y = Y[0, :, :]
    vmin = np.min(solution)
    vmax = np.max(solution)
    fig, ax = plt.subplots()
    for step, t in enumerate(T):
        ax.clear()
        contour = ax.contourf(X, Y, solution[step], cmap="viridis", 
                              vmin=vmin, vmax=vmax)
        ax.set_title(f"Solution at time t = {t:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if step == 0:
            fig.colorbar(contour, ax=ax, label="u(x, y, t)")
        plt.pause(pause_time)
    if show:
        plt.show()
    return fig