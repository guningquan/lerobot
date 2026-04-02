from lerobot.robots.viperx import ViperX, ViperXConfig

config = ViperXConfig(
    port="/dev/ttyDXL_puppet_right",
    id="viperx_right",
    use_degrees=True,
)
robot = ViperX(config)
robot.connect(calibrate=False)

# 检查所有电机的状态
print("Checking all motors status...")
for motor_name, motor in robot.bus.motors.items():
    print(f"\n{'='*60}")
    print(f"Motor: {motor_name} (ID: {motor.id})")
    print(f"{'='*60}")
    
    try:
        # Hardware Error Status
        hw_error = robot.bus.read("Hardware_Error_Status", motor_name, normalize=False)
        print(f"Hardware Error Status: 0x{hw_error:02X} ({'OK' if hw_error == 0 else 'ERROR'})")
        
        # Voltage
        voltage = robot.bus.read("Present_Input_Voltage", motor_name, normalize=False) / 10.0
        print(f"Voltage: {voltage:.1f}V {'✓' if 10.0 <= voltage <= 14.8 else '⚠️'} (正常范围: 10.0-14.8V)")
        
        # Temperature
        temp = robot.bus.read("Present_Temperature", motor_name, normalize=False)
        print(f"Temperature: {temp}°C {'✓' if temp < 70 else '⚠️'} (正常范围: <70°C)")
        
        # Current
        current = robot.bus.read("Present_Current", motor_name, normalize=False)
        print(f"Current: {current}mA")
        
        # Torque Enable
        torque = robot.bus.read("Torque_Enable", motor_name, normalize=False)
        print(f"Torque Enable: {torque}")
        
        # Position
        position = robot.bus.read("Present_Position", motor_name, normalize=False)
        print(f"Present Position: {position}")
        
    except Exception as e:
        print(f"Error reading {motor_name}: {e}")

# 尝试写入 Torque_Enable 测试通信
print(f"\n{'='*60}")
print("Testing Torque_Enable write operation...")
print(f"{'='*60}")

try:
    # 先读取当前值
    current_torque = robot.bus.read("Torque_Enable", "shoulder", normalize=False)
    print(f"Current Torque_Enable for shoulder: {current_torque}")
    
    # 尝试写入（使用更大的重试次数）
    print("Attempting to write Torque_Enable with num_retry=5...")
    robot.bus.write("Torque_Enable", "shoulder", 0, num_retry=5)
    print("✓ Write successful!")
    
    # 验证写入
    new_torque = robot.bus.read("Torque_Enable", "shoulder", normalize=False)
    print(f"New Torque_Enable value: {new_torque}")
    
except Exception as e:
    print(f"✗ Write failed: {e}")

robot.disconnect()
print("\nCheck complete!")