import sys
import numpy as np
import matplotlib.pyplot as plt
from xensesdk import Sensor
import time


def visualize_force(sensor_id="OG000614", realtime=True):
    """
    Display Force distribution in real-time.
    Force: shape=(35, 20, 3) - 3D force distribution
    Shows 3D force vectors at each grid point (35×20 points)
    
    Args:
        sensor_id: Sensor serial number
        realtime: If True, continuously update the display. If False, show single frame.
    """
    sensor = Sensor.create(sensor_id)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'Force Distribution - {sensor_id}\n35×20 Grid Points with 3D Force Vectors', fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.ion()
    
    try:
        while plt.fignum_exists(1):
            force = sensor.selectSensorInfo(Sensor.OutputType.Force)
            
            ax.clear()
            ax.set_title(f'Force Distribution - {sensor_id}\n35×20 Grid Points with 3D Force Vectors', fontsize=14)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            if force is not None:
                if len(force.shape) == 3 and force.shape[2] == 3:
                    h, w = force.shape[:2]  # h=35, w=20
                    
                    # Create grid positions
                    x_grid = np.arange(w)
                    y_grid = np.arange(h)
                    X, Y = np.meshgrid(x_grid, y_grid)
                    Z = np.zeros_like(X)  # Start from z=0 plane
                    
                    # Get force components
                    Fx = force[:, :, 0]
                    Fy = force[:, :, 1]
                    Fz = force[:, :, 2]
                    
                    # Calculate force magnitude for color coding
                    force_magnitude = np.linalg.norm(force, axis=2)
                    
                    # Flatten arrays for quiver plot
                    X_flat = X.flatten()
                    Y_flat = Y.flatten()
                    Z_flat = Z.flatten()
                    Fx_flat = Fx.flatten()
                    Fy_flat = Fy.flatten()
                    Fz_flat = Fz.flatten()
                    magnitude_flat = force_magnitude.flatten()
                    
                    # Plot 3D force vectors using quiver
                    ax.quiver(X_flat, Y_flat, Z_flat, 
                             Fx_flat, Fy_flat, Fz_flat,
                             length=1.0, normalize=True, 
                             colors=plt.cm.hot(magnitude_flat / magnitude_flat.max() if magnitude_flat.max() > 0 else 1),
                             alpha=0.8, arrow_length_ratio=0.3)
                    
                    # Also show grid points
                    ax.scatter(X_flat, Y_flat, Z_flat, 
                              c=magnitude_flat, cmap='hot', s=20, 
                              alpha=0.6, edgecolors='black', linewidths=0.3)
                    
                    # Set equal aspect ratio
                    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), 
                                         max(abs(Fz.max()), abs(Fz.min()))]).max() / 2.0
                    mid_x = (X.max()+X.min()) * 0.5
                    mid_y = (Y.max()+Y.min()) * 0.5
                    mid_z = 0
                    ax.set_xlim(mid_x - max_range, mid_x + max_range)
                    ax.set_ylim(mid_y - max_range, mid_y + max_range)
                    ax.set_zlim(mid_z - max_range, mid_z + max_range)
                    
                else:
                    ax.text(0.5, 0.5, 0.5, f'Invalid Force shape: {force.shape}', 
                            ha='center', va='center')
            else:
                ax.text(0.5, 0.5, 0.5, 'No Force Data', ha='center', va='center')
            
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
    print("Displaying Force distribution in real-time...")
    print("Press Ctrl+C or close the window to exit.")
    
    try:
        visualize_force(sensor_id, realtime=True)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
