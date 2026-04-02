import sys
import numpy as np
import matplotlib.pyplot as plt
from xensesdk import Sensor
import time


def visualize_mesh3dflow(sensor_id="OG000614", realtime=True):
    """
    Display Mesh3DFlow (Mesh deformation vector) in real-time.
    Mesh3DFlow: shape=(35, 20, 3) - Mesh deformation vector
    
    Args:
        sensor_id: Sensor serial number
        realtime: If True, continuously update the display. If False, show single frame.
    """
    sensor = Sensor.create(sensor_id)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'Mesh3DFlow (Deformation) - {sensor_id}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Flow Magnitude')
    
    plt.ion()
    
    try:
        while plt.fignum_exists(1):
            mesh3d_flow = sensor.selectSensorInfo(Sensor.OutputType.Mesh3DFlow)
            
            ax.cla()
            ax.set_title(f'Mesh3DFlow (Deformation) - {sensor_id}\nShape: {mesh3d_flow.shape if mesh3d_flow is not None else "None"}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Flow Magnitude')
            
            if mesh3d_flow is not None:
                if len(mesh3d_flow.shape) == 3 and mesh3d_flow.shape[2] == 3:
                    # Visualize flow as magnitude
                    flow_magnitude = np.linalg.norm(mesh3d_flow, axis=2)
                    h, w = flow_magnitude.shape
                    X, Y = np.meshgrid(np.arange(w), np.arange(h))
                    ax.plot_surface(X, Y, flow_magnitude, cmap='cool', alpha=0.8)
                else:
                    ax.text(0.5, 0.5, 0.5, f'Invalid Mesh3DFlow shape: {mesh3d_flow.shape}', 
                            ha='center', va='center')
            else:
                ax.text(0.5, 0.5, 0.5, 'No Mesh3DFlow Data', ha='center', va='center')
            
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
    print("Displaying Mesh3DFlow in real-time...")
    print("Press Ctrl+C or close the window to exit.")
    
    try:
        visualize_mesh3dflow(sensor_id, realtime=True)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
