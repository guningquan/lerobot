# import ctypes.util
# # Keep original function for other libraries
# _original_find_library = ctypes.util.find_library

# def _patched_find_library(name):
#     # Force pyudev to use the real system libudev
#     if name == "udev":
#         return "/lib/x86_64-linux-gnu/libudev.so.1"
#     return _original_find_library(name)

# # Apply patch before importing pyudev/xensesdk
# ctypes.util.find_library = _patched_find_library

from xensesdk import Sensor
from time import sleep



def main():
    # 1. 创建传感器

    sensor = Sensor.create('OG000635', use_gpu=False ) # OG000635 OG000614
    # sensor = Sensor.create('OG000635') # OG000635 OG000614
    # sensor1 = Sensor.create('OG000635') 

    # 2. 读取传感器数据
    #   sensor.selectSensorInfo 可以通过传入 `Sensor.OutputType` 枚举量获取相应的传感器数据, 顺序或者数量无限制
    #   可选的输出类型参考API说明
    while True:
        # rectify_img, depth= sensor.selectSensorInfo(Sensor.OutputType.Rectify, Sensor.OutputType.Depth)

        rectify, depth, force = sensor.selectSensorInfo(
        Sensor.OutputType.Rectify,
        Sensor.OutputType.Depth,
        Sensor.OutputType.Force
        )

        # 输出数据形状（示例）
        print("校正图像形状:", rectify.shape)       # (700, 400, 3)
        print("深度图像形状:", depth.shape)         # (700, 400)
        print("三维力分布形状:", force.shape)       # (35, 20, 3)
        import cv2
        import numpy as np

        while True:
            rectify, depth, force = sensor.selectSensorInfo(
                Sensor.OutputType.Rectify,
                Sensor.OutputType.Depth,
                Sensor.OutputType.Force
            )

            # 显示校正图像
            cv2.imshow("Rectify Image", rectify)

            # 显示深度图像（归一化到0-255的灰度图）
            depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_display = depth_norm.astype(np.uint8)
            cv2.imshow("Depth Image", depth_display)

            # 可视化三维力分布，每个像素的力强度可通过矢量的范数来表示为灰度图
            force_magnitude = np.linalg.norm(force, axis=2)
            force_norm = cv2.normalize(force_magnitude, None, 0, 255, cv2.NORM_MINMAX)
            force_display = force_norm.astype(np.uint8)
            force_display = cv2.resize(force_display, (rectify.shape[1], rectify.shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Force Magnitude", force_display)

            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:  # 27 is 'esc'
                break

        cv2.destroyAllWindows()
        sensor.release()
        exit()
        # 数据处理
        # ...
        sleep(0.02)

if __name__ == '__main__':
    main()