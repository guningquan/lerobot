from xensesdk import ExampleView
from xensesdk import Sensor
import sys


def main():
    sensor_0 = Sensor.create("OG000614")
    View = ExampleView(sensor_0)
    View2d = View.create2d(Sensor.OutputType.Marker2D,
                Sensor.OutputType.Force,
                Sensor.OutputType.ForceNorm,
                Sensor.OutputType.ForceResultant,)
    
    def callback():
        # diff, depth = sensor_0.selectSensorInfo(Sensor.OutputType.Difference, Sensor.OutputType.Depth)
        rectify, difference, depth, marker2d, force, force_norm, force_resultant, \
            mesh3d, mesh3d_init, mesh3d_flow, timestamp = sensor_0.selectSensorInfo(
                Sensor.OutputType.Rectify,
                Sensor.OutputType.Difference,
                Sensor.OutputType.Depth,
                Sensor.OutputType.Marker2D,
                Sensor.OutputType.Force,
                Sensor.OutputType.ForceNorm,
                Sensor.OutputType.ForceResultant,
                Sensor.OutputType.Mesh3D,
                Sensor.OutputType.Mesh3DInit,
                Sensor.OutputType.Mesh3DFlow,
                Sensor.OutputType.TimeStamp
            )
        View2d.setData(Sensor.OutputType.Marker2D, marker2d)
        View2d.setData(Sensor.OutputType.Force, force)
        View2d.setData(Sensor.OutputType.ForceNorm, force_norm)
        View2d.setData(Sensor.OutputType.ForceResultant, force_resultant)
        View.setDepth(depth)

    View.setCallback(callback)
    View.show()
    sensor_0.release()
    sys.exit()


if __name__ == '__main__':
    main()