import sys
import numpy as np
import matplotlib.pyplot as plt
from xensesdk import Sensor
import cv2
import time


def visualize_rectify(sensor_id="OG000614", realtime=True):
    """
    Display Rectify image in real-time.
    Rectify: shape=(700, 400, 3), BGR format
    
    Args:
        sensor_id: Sensor serial number
        realtime: If True, continuously update the display. If False, show single frame.
    """
    sensor = Sensor.create(sensor_id)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.set_title(f'Rectify Image - {sensor_id}')
    ax.axis('off')
    im = None
    
    plt.ion()
    
    try:
        while plt.fignum_exists(1):
            rectify = sensor.selectSensorInfo(Sensor.OutputType.Rectify)[0]
            
            if rectify is not None:
                # Convert BGR to RGB for display
                rectify_rgb = cv2.cvtColor(rectify, cv2.COLOR_BGR2RGB)
                
                ax.clear()
                ax.axis('off')
                ax.set_title(f'Rectify Image - {sensor_id}\nShape: {rectify.shape}')
                
                if im is None:
                    im = ax.imshow(rectify_rgb)
                else:
                    im.set_data(rectify_rgb)
                
                fig.canvas.draw_idle()
            else:
                ax.text(0.5, 0.5, 'No Rectify Data', ha='center', va='center')
            
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
    print("Displaying Rectify image in real-time...")
    print("Press Ctrl+C or close the window to exit.")
    
    try:
        visualize_rectify(sensor_id, realtime=True)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
