import sys
import numpy as np
import matplotlib.pyplot as plt
from xensesdk import Sensor
import time


def visualize_forcenorm(sensor_id="OG000614", realtime=True):
    """
    Display ForceNorm (Normal force component) in real-time.
    ForceNorm: shape=(35, 20, 3) - Normal force component
    Shows 3D normal force vectors at each grid point (35×20 points)
    
    Args:
        sensor_id: Sensor serial number
        realtime: If True, continuously update the display. If False, show single frame.
    """
    sensor = Sensor.create(sensor_id)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'ForceNorm (Normal) - {sensor_id}\n35×20 Grid Points with 3D Normal Force Vectors', fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.ion()
    
    try:
        while plt.fignum_exists(1):
            force_norm = sensor.selectSensorInfo(Sensor.OutputType.ForceNorm)
            
            ax.clear()
            ax.set_title(f'ForceNorm (Normal) - {sensor_id}\n35×20 Grid Points with 3D Normal Force Vectors', fontsize=14)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            if force_norm is not None:
                if len(force_norm.shape) == 3 and force_norm.shape[2] == 3:
                    h, w = force_norm.shape[:2]  # h=35, w=20
                    
                    # Create grid positions
                    x_grid = np.arange(w)
                    y_grid = np.arange(h)
                    X, Y = np.meshgrid(x_grid, y_grid)
                    Z = np.zeros_like(X)  # Start from z=0 plane
                    
                    # Get normal force components
                    Fx = force_norm[:, :, 0]
                    Fy = force_norm[:, :, 1]
                    Fz = force_norm[:, :, 2]
                    
                    # Calculate normal force magnitude for color coding
                    force_norm_magnitude = np.linalg.norm(force_norm, axis=2)
                    
                    # Flatten arrays for quiver plot
                    X_flat = X.flatten()
                    Y_flat = Y.flatten()
                    Z_flat = Z.flatten()
                    Fx_flat = Fx.flatten()
                    Fy_flat = Fy.flatten()
                    Fz_flat = Fz.flatten()
                    magnitude_flat = force_norm_magnitude.flatten()
                    
                    # Plot 3D normal force vectors using quiver
                    ax.quiver(X_flat, Y_flat, Z_flat, 
                             Fx_flat, Fy_flat, Fz_flat,
                             length=1.0, normalize=True, 
                             colors=plt.cm.coolwarm(magnitude_flat / magnitude_flat.max() if magnitude_flat.max() > 0 else 1),
                             alpha=0.8, arrow_length_ratio=0.3)
                    
                    # Also show grid points
                    ax.scatter(X_flat, Y_flat, Z_flat, 
                              c=magnitude_flat, cmap='coolwarm', s=20, 
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
                    ax.text(0.5, 0.5, 0.5, f'Invalid ForceNorm shape: {force_norm.shape}', 
                            ha='center', va='center')
            else:
                ax.text(0.5, 0.5, 0.5, 'No ForceNorm Data', ha='center', va='center')
            
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
    print("Displaying ForceNorm in real-time...")
    print("Press Ctrl+C or close the window to exit.")
    
    try:
        visualize_forcenorm(sensor_id, realtime=True)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
