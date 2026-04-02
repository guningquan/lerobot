import cv2
import signal

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

# Check if OpenCV has GUI support
def has_gui_support():
    """Check if OpenCV has GUI support (imshow, waitKey, etc.)"""
    try:
        # Try to create a test window
        test_img = cv2.namedWindow("__test__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__test__")
        return True
    except cv2.error:
        return False

# Construct an `OpenCVCameraConfig` with your desired FPS, resolution, color mode, and rotation.
config = OpenCVCameraConfig(
    # index_or_path="/dev/CAM_HIGH",
    index_or_path="/dev/CAM_LEFT_WRIST",
    fps=30,
    width=640,
    height=480,
    color_mode=ColorMode.RGB,
    rotation=Cv2Rotation.NO_ROTATION,
    fourcc="YUYV"
)

# Instantiate and connect an `OpenCVCamera`, performing a warm-up read (default).
camera = OpenCVCamera(config)
camera.connect()

# Check GUI support
gui_available = has_gui_support()
if not gui_available:
    print("Warning: OpenCV GUI support not available (opencv-python-headless detected).")
    print("Video will not be displayed. Press Ctrl+C to stop.")

# Signal handler for graceful shutdown
shutdown = False

def signal_handler(sig, frame):
    global shutdown
    print("\nShutdown signal received. Stopping...")
    shutdown = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Read frames asynchronously in a loop and display them until user presses 'q' or ESC
window_name = "Camera Preview"
try:
    if gui_available:
        print("Press 'q' or ESC to quit...")
    else:
        print("Press Ctrl+C to quit...")
    
    frame_count = 0
    while not shutdown:
        frame = camera.async_read(timeout_ms=200)
        frame_count += 1
        
        if gui_available:
            # Convert RGB to BGR for OpenCV display (if color_mode is RGB)
            if config.color_mode == ColorMode.RGB:
                display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                display_frame = frame
            
            # Display the frame
            try:
                cv2.imshow(window_name, display_frame)
                
                # Check for exit key (q or ESC)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    print(f"Exiting after {frame_count} frames")
                    break
            except cv2.error as e:
                print(f"Error displaying frame: {e}")
                print("Falling back to headless mode...")
                gui_available = False
        
        # Print frame info periodically
        if frame_count % 30 == 0:
            print(f"Frame {frame_count} shape: {frame.shape}")
finally:
    camera.disconnect()
    if gui_available:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # Ignore errors if GUI is not available
    print("Camera disconnected and windows closed.")