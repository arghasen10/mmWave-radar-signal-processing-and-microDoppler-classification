import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_rangeDop(Dopdata_sum, rng_grid, vel_grid):
    # Create figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Create meshgrid for plotting
    vel_grid, rng_grid = np.meshgrid(vel_grid, rng_grid)

    # Plot surface
    ax.plot_surface(vel_grid, rng_grid, Dopdata_sum, cmap='viridis')

    # Set labels and title
    ax.set_xlabel('Doppler Velocity (m/s)')
    ax.set_ylabel('Range (meters)')
    ax.set_zlabel('Power')
    plt.title('Range-Doppler heatmap')

    # Set colorbar
    cbar = plt.colorbar()
    cbar.set_label('Power')

    plt.show()

# Example usage
# Assuming Dopdata_sum, rng_grid, and vel_grid are defined
# plot_rangeDop(Dopdata_sum, rng_grid, vel_grid)
