import sys
import numpy as np
import matplotlib.pyplot as plt
from xensesdk import Sensor
import time


def visualize_timestamp(sensor_id="OG000614", realtime=True):
    """
    Display TimeStamp information in real-time.
    TimeStamp: float - Sensor timestamp, unit: s
    
    Args:
        sensor_id: Sensor serial number
        realtime: If True, continuously update the display. If False, show single frame.
    """
    sensor = Sensor.create(sensor_id)
    
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    plt.ion()
    
    try:
        while plt.fignum_exists(1):
            timestamp = sensor.selectSensorInfo(Sensor.OutputType.TimeStamp)
            
            ax.clear()
            ax.axis('off')
            
            if timestamp is not None:
                info_text = f"TimeStamp Information - {sensor_id}\n\n"
                info_text += f"TimeStamp: {timestamp:.6f} s\n"
                info_text += f"TimeStamp: {timestamp * 1000:.3f} ms\n"
                info_text += f"TimeStamp: {timestamp * 1000000:.0f} μs\n\n"
                info_text += f"Type: {type(timestamp).__name__}\n"
                info_text += f"Value: {timestamp}"
                
                ax.text(0.1, 0.5, info_text, fontsize=12, family='monospace',
                       verticalalignment='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'No TimeStamp Data', ha='center', va='center')
            
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
    print("Displaying TimeStamp information in real-time...")
    print("Press Ctrl+C or close the window to exit.")
    
    try:
        visualize_timestamp(sensor_id, realtime=True)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
