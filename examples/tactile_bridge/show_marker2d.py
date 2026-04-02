import sys
import numpy as np
import matplotlib.pyplot as plt
from xensesdk import Sensor
import time


def visualize_marker2d(sensor_id="OG000614", realtime=True):
    """
    Display Marker2D displacement in real-time.
    Marker2D: shape=(26, 14, 2) - Tangential displacement
    Shows the movement of 26×14 points with initial positions, displacement vectors, and magnitude.
    
    Args:
        sensor_id: Sensor serial number
        realtime: If True, continuously update the display. If False, show single frame.
    """
    sensor = Sensor.create(sensor_id)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.set_title(f'Marker2D Displacement - {sensor_id}\n26×14 Points Movement', fontsize=14)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.grid(True, alpha=0.3)
    
    plt.ion()
    
    # Store initial positions (first frame)
    initial_positions = None
    
    try:
        while plt.fignum_exists(1):
            marker2d = sensor.selectSensorInfo(Sensor.OutputType.Marker2D)
            ax.clear()
            ax.set_title(f'Marker2D Displacement - {sensor_id}\n26×14 Points Movement', fontsize=14)
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.grid(True, alpha=0.3)

            if marker2d is not None:
                if len(marker2d.shape) == 3 and marker2d.shape[2] == 2:
                    h, w = marker2d.shape[:2]  # h=26, w=14
                    
                    # Initialize reference positions on first frame
                    if initial_positions is None:
                        # Create grid of initial positions
                        x_grid = np.arange(w)
                        y_grid = np.arange(h)
                        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
                        initial_positions = np.stack([X_grid, Y_grid], axis=2)
                    
                    # Calculate displacement magnitude for color coding
                    displacement_magnitude = np.linalg.norm(marker2d, axis=2)
                    
                    # Get displacement components
                    U = marker2d[:, :, 0]  # X displacement
                    V = marker2d[:, :, 1]  # Y displacement
                    
                    # Calculate current positions (initial + displacement)
                    current_X = initial_positions[:, :, 0] + U
                    current_Y = initial_positions[:, :, 1] + V
                    
                    # Flatten arrays for plotting
                    initial_X_flat = initial_positions[:, :, 0].flatten()
                    initial_Y_flat = initial_positions[:, :, 1].flatten()
                    current_X_flat = current_X.flatten()
                    current_Y_flat = current_Y.flatten()
                    U_flat = U.flatten()
                    V_flat = V.flatten()
                    magnitude_flat = displacement_magnitude.flatten()
                    
                    # Plot 1: Initial positions (grid points) - gray dots
                    ax.scatter(initial_X_flat, initial_Y_flat, c='lightgray', s=30, 
                              marker='o', alpha=0.5, label='Initial Positions', zorder=1)
                    
                    # Plot 2: Current positions - colored by displacement magnitude
                    ax.scatter(current_X_flat, current_Y_flat, c=magnitude_flat, 
                              s=50, cmap='hot', marker='o', 
                              edgecolors='black', linewidths=0.5,
                              zorder=3, vmin=0)
                    
                    # Plot 3: Displacement vectors (arrows) from initial to current
                    ax.quiver(initial_X_flat, initial_Y_flat, U_flat, V_flat, 
                             magnitude_flat, angles='xy', scale_units='xy', scale=1,
                             cmap='viridis', width=0.003, alpha=0.6, zorder=2)
                    
                    ax.set_aspect('equal')
                    
                else:
                    ax.text(0.5, 0.5, f'Invalid Marker2D shape: {marker2d.shape}', 
                            ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'No Marker2D Data', ha='center', va='center')
            
            fig.canvas.draw_idle()
            
            plt.pause(0.01)
            if not realtime:
                break
            time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\nStopping visualization...")
    finally:
        plt.ioff()
        sensor.release()


def main():
    sensor_id = "OG000614"
    if len(sys.argv) > 1:
        sensor_id = sys.argv[1]
    
    print(f"Connecting to sensor: {sensor_id}")
    print("Displaying Marker2D displacement in real-time...")
    print("Press Ctrl+C or close the window to exit.")
    
    try:
        visualize_marker2d(sensor_id, realtime=True)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
