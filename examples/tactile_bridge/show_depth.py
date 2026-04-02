import sys
import numpy as np
import matplotlib.pyplot as plt
from xensesdk import Sensor
import time


def visualize_depth(sensor_id="OG000614", realtime=True):
    """
    Display Depth image in real-time.
    Depth: shape=(700, 400), unit: mm
    
    Args:
        sensor_id: Sensor serial number
        realtime: If True, continuously update the display. If False, show single frame.
    """
    sensor = Sensor.create(sensor_id)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.set_title(f'Depth Image (Unit: mm) - {sensor_id}')
    ax.axis('off')
    im = None
    cbar = None
    
    plt.ion()
    
    try:
        while plt.fignum_exists(1):
            depth = sensor.selectSensorInfo(Sensor.OutputType.Depth)[0]
            
            if depth is not None:
                ax.clear()
                ax.axis('off')
                ax.set_title(f'Depth Image (Unit: mm) - {sensor_id}\nShape: {depth.shape}')
                
                if im is None:
                    im = ax.imshow(depth, cmap='jet')
                    cbar = plt.colorbar(im, ax=ax, label='Depth (mm)')
                else:
                    im.set_array(depth)
                    im.set_clim(vmin=depth.min(), vmax=depth.max())
                
                fig.canvas.draw_idle()
            else:
                ax.text(0.5, 0.5, 'No Depth Data', ha='center', va='center')
            
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
    print("Displaying Depth image in real-time...")
    print("Press Ctrl+C or close the window to exit.")
    
    try:
        visualize_depth(sensor_id, realtime=True)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
