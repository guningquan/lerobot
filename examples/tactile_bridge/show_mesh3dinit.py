import sys
import numpy as np
import matplotlib.pyplot as plt
from xensesdk import Sensor
import time


def visualize_mesh3dinit(sensor_id="OG000614", realtime=True):
    """
    Display Mesh3DInit (Initial 3D mesh) in real-time.
    Mesh3DInit: shape=(35, 20, 3) - Initial 3D mesh
    
    Args:
        sensor_id: Sensor serial number
        realtime: If True, continuously update the display. If False, show single frame.
    """
    sensor = Sensor.create(sensor_id)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'Mesh3DInit (Initial) - {sensor_id}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.ion()
    
    try:
        while plt.fignum_exists(1):
            mesh3d_init = sensor.selectSensorInfo(Sensor.OutputType.Mesh3DInit)
            
            ax.cla()
            ax.set_title(f'Mesh3DInit (Initial) - {sensor_id}\nShape: {mesh3d_init.shape if mesh3d_init is not None else "None"}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            if mesh3d_init is not None:
                if len(mesh3d_init.shape) == 3 and mesh3d_init.shape[2] == 3:
                    h, w = mesh3d_init.shape[:2]
                    X, Y = np.meshgrid(np.arange(w), np.arange(h))
                    ax.plot_surface(X, Y, mesh3d_init[:, :, 2], cmap='plasma', alpha=0.8)
                else:
                    ax.text(0.5, 0.5, 0.5, f'Invalid Mesh3DInit shape: {mesh3d_init.shape}', 
                            ha='center', va='center')
            else:
                ax.text(0.5, 0.5, 0.5, 'No Mesh3DInit Data', ha='center', va='center')
            
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
    print("Displaying Mesh3DInit in real-time...")
    print("Press Ctrl+C or close the window to exit.")
    
    try:
        visualize_mesh3dinit(sensor_id, realtime=True)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
