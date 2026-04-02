import sys
import numpy as np
import matplotlib.pyplot as plt
from xensesdk import Sensor
import time


def visualize_sensor_data(sensor_id="OG000614", realtime=True):
    """
    Display Marker2D, Force, ForceNorm, and ForceResultant in real-time.
    
    Args:
        sensor_id: Sensor serial number
        realtime: If True, continuously update the display. If False, show single frame.
    """
    # Create sensor
    sensor = Sensor.create(sensor_id)
    
    # Create separate figures for each data type
    # Marker2D: shape=(26, 14, 2) - Tangential displacement
    fig1 = plt.figure(1, figsize=(8, 6))
    ax1 = fig1.add_subplot(111)
    ax1.set_title(f'Marker2D Displacement - {sensor_id}')
    ax1.axis('off')
    
    # Force: shape=(35, 20, 3) - 3D force distribution
    fig2 = plt.figure(2, figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    ax2.set_title(f'Force Distribution - {sensor_id}')
    ax2.axis('off')
    im2 = None
    cbar2 = None
    
    # ForceNorm: shape=(35, 20, 3) - Normal force component
    fig3 = plt.figure(3, figsize=(8, 6))
    ax3 = fig3.add_subplot(111)
    ax3.set_title(f'ForceNorm (Normal) - {sensor_id}')
    ax3.axis('off')
    im3 = None
    cbar3 = None
    
    # ForceResultant: shape=(6,) - 6D resultant force
    fig4 = plt.figure(4, figsize=(8, 6))
    ax4 = fig4.add_subplot(111)
    ax4.set_title(f'ForceResultant (6D) - {sensor_id}')
    ax4.set_ylabel('Force/Moment')
    ax4.grid(True, alpha=0.3)
    
    plt.ion()  # Turn on interactive mode
    
    # Track if format has been checked
    format_checked = False
    
    try:
        while any(plt.fignum_exists(i) for i in range(1, 5)):
            # Get only the required sensor data
            marker2d, force, force_norm, force_resultant = sensor.selectSensorInfo(
                Sensor.OutputType.Marker2D,
                Sensor.OutputType.Force,
                Sensor.OutputType.ForceNorm,
                Sensor.OutputType.ForceResultant
            )
            
            # Check and print data formats once for verification
            if not format_checked:
                print("\n=== Data Format Verification ===")
                if marker2d is not None:
                    print(f"Marker2D: shape={marker2d.shape}, dtype={marker2d.dtype}, expected=(26, 14, 2)")
                else:
                    print("Marker2D: None")
                if force is not None:
                    print(f"Force: shape={force.shape}, dtype={force.dtype}, expected=(35, 20, 3)")
                else:
                    print("Force: None")
                if force_norm is not None:
                    print(f"ForceNorm: shape={force_norm.shape}, dtype={force_norm.dtype}, expected=(35, 20, 3)")
                else:
                    print("ForceNorm: None")
                if force_resultant is not None:
                    print(f"ForceResultant: shape={force_resultant.shape}, dtype={force_resultant.dtype}, expected=(6,), values={force_resultant}")
                else:
                    print("ForceResultant: None")
                print("================================\n")
                format_checked = True
            
            # Update 1. Marker2D (26, 14, 2) - Tangential displacement
            if plt.fignum_exists(1):
                ax1.clear()
                ax1.axis('off')
                ax1.set_title(f'Marker2D Displacement - {sensor_id}')
                if marker2d is not None:
                    # Verify shape
                    if len(marker2d.shape) == 3 and marker2d.shape[2] == 2:
                        h, w = marker2d.shape[:2]
                        x = np.arange(w)
                        y = np.arange(h)
                        X, Y = np.meshgrid(x, y)
                        U = marker2d[:, :, 0]
                        V = marker2d[:, :, 1]
                        ax1.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
                        ax1.set_aspect('equal')
                    else:
                        ax1.text(0.5, 0.5, f'Invalid Marker2D shape: {marker2d.shape}', 
                                ha='center', va='center', transform=ax1.transAxes)
                else:
                    ax1.text(0.5, 0.5, 'No Marker2D Data', ha='center', va='center')
                fig1.canvas.draw_idle()
            
            # Update 2. Force (35, 20, 3) - 3D force distribution
            if plt.fignum_exists(2):
                ax2.clear()
                ax2.axis('off')
                ax2.set_title(f'Force Distribution - {sensor_id}')
                if force is not None:
                    # Verify shape
                    if len(force.shape) == 3 and force.shape[2] == 3:
                        # Calculate force magnitude
                        force_magnitude = np.linalg.norm(force, axis=2)
                        if im2 is None:
                            im2 = ax2.imshow(force_magnitude, cmap='hot')
                            cbar2 = plt.colorbar(im2, ax=ax2, label='Force Magnitude')
                        else:
                            im2.set_array(force_magnitude)
                            im2.set_clim(vmin=force_magnitude.min(), vmax=force_magnitude.max())
                    else:
                        ax2.text(0.5, 0.5, f'Invalid Force shape: {force.shape}', 
                                ha='center', va='center', transform=ax2.transAxes)
                else:
                    ax2.text(0.5, 0.5, 'No Force Data', ha='center', va='center')
                fig2.canvas.draw_idle()
            
            # Update 3. ForceNorm (35, 20, 3) - Normal force component
            if plt.fignum_exists(3):
                ax3.clear()
                ax3.axis('off')
                ax3.set_title(f'ForceNorm (Normal) - {sensor_id}')
                if force_norm is not None:
                    # Verify shape
                    if len(force_norm.shape) == 3 and force_norm.shape[2] == 3:
                        # Calculate normal force magnitude
                        force_norm_magnitude = np.linalg.norm(force_norm, axis=2)
                        if im3 is None:
                            im3 = ax3.imshow(force_norm_magnitude, cmap='coolwarm')
                            cbar3 = plt.colorbar(im3, ax=ax3, label='Normal Force Magnitude')
                        else:
                            im3.set_array(force_norm_magnitude)
                            im3.set_clim(vmin=force_norm_magnitude.min(), vmax=force_norm_magnitude.max())
                    else:
                        ax3.text(0.5, 0.5, f'Invalid ForceNorm shape: {force_norm.shape}', 
                                ha='center', va='center', transform=ax3.transAxes)
                else:
                    ax3.text(0.5, 0.5, 'No ForceNorm Data', ha='center', va='center')
                fig3.canvas.draw_idle()
            
            # Update 4. ForceResultant (6,) - 6D resultant force
            if plt.fignum_exists(4):
                ax4.clear()
                ax4.set_title(f'ForceResultant (6D) - {sensor_id}')
                ax4.set_ylabel('Force/Moment')
                ax4.grid(True, alpha=0.3)
                if force_resultant is not None:
                    # Verify shape
                    if len(force_resultant.shape) == 1 and force_resultant.shape[0] == 6:
                        labels = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
                        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
                        bars = ax4.bar(labels, force_resultant, color=colors)
                        for bar, val in zip(bars, force_resultant):
                            height = bar.get_height()
                            ax4.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{val:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
                    else:
                        ax4.text(0.5, 0.5, f'Invalid ForceResultant shape: {force_resultant.shape}', 
                                ha='center', va='center', transform=ax4.transAxes)
                else:
                    ax4.text(0.5, 0.5, 'No ForceResultant Data', ha='center', va='center')
                fig4.canvas.draw_idle()
            
            plt.pause(0.01)  # Small pause for GUI update
            if not realtime:
                break
            time.sleep(0.01)  # Control update rate
                
    except KeyboardInterrupt:
        print("\nStopping visualization...")
    finally:
        plt.ioff()  # Turn off interactive mode
        sensor.release()
            
            
def main():
    """
    Main function to run the visualization.
    You can modify the sensor_id here or pass it as command line argument.
    Usage:
        python analyze_all_data.py [sensor_id]
    """
    sensor_id = "OG000614"  # Default sensor ID
    
    # Check if sensor ID is provided as command line argument
    if len(sys.argv) > 1:
        sensor_id = sys.argv[1]
    
    print(f"Connecting to sensor: {sensor_id}")
    print("Displaying real-time data for Marker2D, Force, ForceNorm, and ForceResultant...")
    print("Press Ctrl+C or close the matplotlib windows to exit.")
    
    try:
        visualize_sensor_data(sensor_id, realtime=True)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
