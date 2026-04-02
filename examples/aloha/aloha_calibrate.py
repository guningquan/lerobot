from lerobot.robots.viperx import ViperX, ViperXConfig
from lerobot.teleoperators.widowx import WidowX, WidowXConfig
# ------------------------------------------------------------

config_follower_right = ViperXConfig(
    port="/dev/ttyDXL_puppet_right",
    id="viperx_right",
)

follower_right = ViperX(config_follower_right)
follower_right.connect(calibrate=False)
follower_right.calibrate()
follower_right.disconnect()

# ------------------------------------------------------------

# config_leader_right = WidowXConfig(
#     port="/dev/ttyDXL_master_right",
#     id="widowx_right",
# )

# leader_right = WidowX(config_leader_right)
# leader_right.connect(calibrate=False)
# leader_right.calibrate()
# leader_right.disconnect()

# ------------------------------------------------------------

# config_follower_left = ViperXConfig(
#     port="/dev/ttyDXL_puppet_left",
#     id="viperx_left",
# )

# follower_left = ViperX(config_follower_left)
# follower_left.connect(calibrate=False)
# follower_left.calibrate()
# follower_left.disconnect()

# ------------------------------------------------------------

# config_leader_left = WidowXConfig(
#     port="/dev/ttyDXL_master_left",
#     id="widowx_left",
# )

# leader_left = WidowX(config_leader_left)
# leader_left.connect(calibrate=False)
# leader_left.calibrate()
# leader_left.disconnect()
