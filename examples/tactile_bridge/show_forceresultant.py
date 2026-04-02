import sys
import numpy as np
import matplotlib.pyplot as plt
from xensesdk import Sensor
import time


def visualize_forceresultant(sensor_id="OG000614", realtime=True):
    """
    Display ForceResultant (6D resultant force) in real-time.
    ForceResultant: shape=(6,) - 6D resultant force
    
    Args:
        sensor_id: Sensor serial number
        realtime: If True, continuously update the display. If False, show single frame.
    """
    sensor = Sensor.create(sensor_id)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_title(f'ForceResultant (6D) - {sensor_id}')
    ax.set_ylabel('Force/Moment')
    ax.grid(True, alpha=0.3)
    
    plt.ion()
    
    try:
        while plt.fignum_exists(1):
            force_resultant = sensor.selectSensorInfo(Sensor.OutputType.ForceResultant)
            
            ax.clear()
            ax.set_title(f'ForceResultant (6D) - {sensor_id}\nShape: {force_resultant.shape if force_resultant is not None else "None"}')
            ax.set_ylabel('Force/Moment')
            ax.grid(True, alpha=0.3)
            
            if force_resultant is not None:
                if len(force_resultant.shape) == 1 and force_resultant.shape[0] == 6:
                    labels = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
                    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
                    bars = ax.bar(labels, force_resultant, color=colors)
                    for bar, val in zip(bars, force_resultant):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                f'{val:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
                else:
                    ax.text(0.5, 0.5, f'Invalid ForceResultant shape: {force_resultant.shape}', 
                            ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'No ForceResultant Data', ha='center', va='center')
            
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
    print("Displaying ForceResultant in real-time...")
    print("Press Ctrl+C or close the window to exit.")
    
    try:
        visualize_forceresultant(sensor_id, realtime=True)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
