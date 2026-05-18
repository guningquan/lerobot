### Task parameters
import os

if os.getlogin() == 'guningquan':
    DATA_DIR = '/mnt/ssd1/guningquan/Programs_server/act_dataset_checkpoint/dataset'
elif os.getlogin() == 'theodoreliu':
    DATA_DIR = '/home/theodoreliu/Dataset_and_Checkpoint/dataset'
elif os.getlogin() == 'ubuntu20' or os.getlogin() == 'ubuntu22':
    DATA_DIR = '/home/robot/Dataset_and_Checkpoint/dataset'
else:
    raise ValueError(f"Unknown user: {os.getlogin()}")


TASK_CONFIGS = {

    'velcro_taping_act_3cams': {
        'dataset_dir': DATA_DIR + '/velcro_taping',
        'num_episodes': 50,
        'episode_len': 1000,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist']
    },
    'velcro_taping_3cams': {
        'dataset_dir': DATA_DIR + '/velcro_taping',
        'num_episodes': 50,
        'episode_len': 1000,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel']
    },

    'zip_tie_cotrain':{
            'dataset_dir': [
                DATA_DIR + '/zip_tie',
                # DATA_DIR + '/mobile_aloha/aloha_mobile_wash_pan',
                DATA_DIR + '/mobile_aloha/aloha_static_cotraining_datasets',
                # DATA_DIR + '/velcro_taping',
            ], # only the first dataset_dir is used for val
            'stats_dir': [
                DATA_DIR + '/zip_tie',
            ],
            'sample_weights': [7.5, 2.5],
            'train_ratio': 0.9, # ratio of train data from the first dataset_dir
            'episode_len': 1100,
            'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel'
                         ]
        },

    'pre-train-3camera': {
        'dataset_dir': [
            DATA_DIR + '/mobile_aloha/aloha_static_cotraining_datasets',
        ],
        'train_ratio': 0.95,
        'episode_len': 1000,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         ]
    },

    'pre-train-4camera': {
        'dataset_dir': [
            DATA_DIR + '/mobile_aloha/aloha_static_cotraining_datasets',
        ],
        'train_ratio': 0.95,
        'episode_len': 1000,
        'camera_names': ['cam_high',
                         'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         ]
    },

    'zip_tie_data_collection': {
        'dataset_dir': DATA_DIR + '/zip_tie',
        'episode_len': 900,
        'camera_names': ['cam_high',
                         'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel'
                         ]
    },
    'zip_tie_gel_unet': {
        'dataset_dir': DATA_DIR + '/zip_tie',
        'episode_len': 900,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel'
                         ]
    },

    'zip_tie_gel_resnet': {
        'dataset_dir': DATA_DIR + '/zip_tie',
        'episode_len': 900,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel'
                         ]
    },

    'zip_tie_gel_act': {
        'dataset_dir': DATA_DIR + '/zip_tie',
        'episode_len': 900,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel'
                         ]
    },
    'zip_tie_act': {
        'dataset_dir': DATA_DIR + '/zip_tie',
        'episode_len': 900,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         # 'gel'
                         ]
    },
    'zip_tie_gel_cbam': {
        'dataset_dir': DATA_DIR + '/zip_tie',
        'episode_len': 900,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel'
                         ]
    },

    'velcro_taping_data_collection': {
        'dataset_dir': DATA_DIR + '/velcro_taping',
        'episode_len': 1100,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist', 'gel']
    },

    'velcro_taping_data_collection_test': {
        'dataset_dir': DATA_DIR + '/velcro_taping_test',
        'episode_len': 1000,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist', 'cam_low']
    },

    'velcro_taping_gel_act': {
        'dataset_dir': DATA_DIR + '/velcro_taping',
        'episode_len': 1100,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel'
                         ]
    },
    'velcro_taping_act': {
        'dataset_dir': DATA_DIR + '/velcro_taping',
        'episode_len': 1100,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         # 'gel'
                         ]
    },
    'velcro_taping_gel_tactile_resnet': {
        'dataset_dir': DATA_DIR + '/velcro_taping',
        'episode_len': 1100,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel'
                         ]
    },
    'velcro_taping_gel_tactile_unet': {
        'dataset_dir': DATA_DIR + '/velcro_taping',
        'episode_len': 1100,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel'
                         ]
    },
    'velcro_taping_gel_tactile_cbam': {
        'dataset_dir': DATA_DIR + '/velcro_taping',
        'episode_len': 1100,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel'
                         ]
    },
    'zip_tie_gel_act_front': {
        'dataset_dir': DATA_DIR + '/zip_tie/front',
        'episode_len': 900,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel'
                         ]
    },
    'zip_tie_gel_tactile_resnet_front': {
        'dataset_dir': DATA_DIR + '/zip_tie/front',
        'episode_len': 900,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel'
                         ]
    },
    'zip_tie_gel_tactile_unet_front': {
        'dataset_dir': DATA_DIR + '/zip_tie/front',
        'episode_len': 900,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel'
                         ]
    },
    'velcro_taping_gel_act_cotrain':{
            'dataset_dir': [
                DATA_DIR + '/velcro_taping',
                # DATA_DIR + '/mobile_aloha/aloha_mobile_wash_pan',
                DATA_DIR + '/mobile_aloha',
                # DATA_DIR + '/velcro_taping',
            ], # only the first dataset_dir is used for val
            'stats_dir': [
                DATA_DIR + '/velcro_taping',
            ],
            'sample_weights': [7.5, 2.5],
            'train_ratio': 0.9, # ratio of train data from the first dataset_dir
            'episode_len': 1100,
            'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel'
                         ]
    },

    'zip_tie_random_data_collection': {
        'dataset_dir': DATA_DIR + '/zip_tie_random',
        'episode_len': 900,
        'camera_names': ['cam_high',
                        # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel'
                         ]
    },

    'zip_tie_random_act_notactile': {
        'dataset_dir': DATA_DIR + '/zip_tie_random',
        'episode_len': 900,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         # 'gel'
                         ]
    },
    'zip_tie_random_act_tactile': {
        'dataset_dir': DATA_DIR + '/zip_tie_random',
        'episode_len': 900,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel'
                         ]
    },

    'zip_tie_random_unet_cotrain': {  # include loss
        'dataset_dir': [
            DATA_DIR + '/zip_tie_random',
            # DATA_DIR + '/mobile_aloha/aloha_mobile_wash_pan',
            DATA_DIR + '/mobile_aloha',
            # DATA_DIR + '/velcro_taping',
        ],  # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/zip_tie_random',
        ],
        'sample_weights': [7.5, 2.5],
        'train_ratio': 0.95,  # ratio of train data from the first dataset_dir
        'episode_len': 900,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel'
                         ]
    },

    'zip_tie_random_diffusion_notactile': {
        'dataset_dir': DATA_DIR + '/zip_tie_random',
        'episode_len': 900,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         # 'gel'
                         ]
    },


    'zip_tie_random_unet_cotrain_noloss': {
            'dataset_dir': [
                DATA_DIR + '/zip_tie_random',
                # DATA_DIR + '/mobile_aloha/aloha_mobile_wash_pan',
                DATA_DIR + '/mobile_aloha',
                # DATA_DIR + '/velcro_taping',
            ],  # only the first dataset_dir is used for val
            'stats_dir': [
                DATA_DIR + '/zip_tie_random',
            ],
            'sample_weights': [7.5, 2.5],
            'train_ratio': 0.95,  # ratio of train data from the first dataset_dir
            'episode_len': 900,
            'camera_names': ['cam_high',
                             # 'cam_low',
                             'cam_left_wrist',
                             'cam_right_wrist',
                             'gel'
                             ]
        },


    'zip_tie_random_act_cotrain': {
        'dataset_dir': [
            DATA_DIR + '/zip_tie_random',
            # DATA_DIR + '/mobile_aloha/aloha_mobile_wash_pan',
            DATA_DIR + '/mobile_aloha',
            # DATA_DIR + '/velcro_taping',
        ],  # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/zip_tie_random',
        ],
        'sample_weights': [7.5, 2.5],
        'train_ratio': 0.95,  # ratio of train data from the first dataset_dir
        'episode_len': 900,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel'
                         ]
    },

    'zip_tie_random_resnet_loss': {
        'dataset_dir': DATA_DIR + '/zip_tie_random',
        'episode_len': 900,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel'
                         ]
    },

    'zip_tie_random_unet_loss': {
        'dataset_dir': DATA_DIR + '/zip_tie_random',
        'episode_len': 900,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel'
                         ]
    },

    'zip_tie_random_resnet_cotrain_loss': {
        'dataset_dir': [
            DATA_DIR + '/zip_tie_random',
            # DATA_DIR + '/mobile_aloha/aloha_mobile_wash_pan',
            DATA_DIR + '/mobile_aloha',
            # DATA_DIR + '/velcro_taping',
        ],  # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/zip_tie_random',
        ],
        'sample_weights': [7.5, 2.5],
        'train_ratio': 0.95,  # ratio of train data from the first dataset_dir
        'episode_len': 900,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel'
                         ]
    },

    'zip_tie_random_unet_lowcotrain': {  # include loss
        'dataset_dir': [
            DATA_DIR + '/zip_tie_random',
            # DATA_DIR + '/mobile_aloha/aloha_mobile_wash_pan',
            DATA_DIR + '/mobile_aloha',
            # DATA_DIR + '/velcro_taping',
        ],  # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/zip_tie_random',
        ],
        'sample_weights': [9.0, 1.0],
        'train_ratio': 0.99,  # ratio of train data from the first dataset_dir
        'episode_len': 900,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel'
                         ]
    },
    'velcro_taping_gel_tactile_unet_lowloss': {
        'dataset_dir': DATA_DIR + '/velcro_taping',
        'episode_len': 1100,
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         'gel'
                         ]
    },
    'boxlock_act': {
        'dataset_dir': DATA_DIR + '/boxlock',
        'episode_len': 900,  # 900
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         ]
    },
    'alarm_act': {
        'dataset_dir': DATA_DIR + '/alarm',
        'episode_len': 900,  # 900
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         ]
    },
    'alarm_act_4cams': {
        'dataset_dir': DATA_DIR + '/alarm',
        'episode_len': 900,  # 900
        'camera_names': ['cam_high',
                         'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         ]
    },

    'alarm_act_3cams': {
        'dataset_dir': DATA_DIR + '/alarm_random_pos',
        'episode_len': 900,  # 900
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         ]
    },

    'boxlockdown_act': {
        'dataset_dir': DATA_DIR + '/boxlockdown',
        'episode_len': 750,  # 900
        'camera_names': ['cam_high',
                         'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         ]
    },
    'boxlockdown_act_3cam': {
        'dataset_dir': DATA_DIR + '/boxlockdown',
        'episode_len': 750,  # 900
        'camera_names': ['cam_high',
                         # 'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         ]
    },

    'alarm_bias_act_4cams': {
        'dataset_dir': DATA_DIR + '/alarm_bias',
        'episode_len': 1000,  # 900
        'camera_names': ['cam_high',
                         'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         ]
    },

    'alarm_random_pos_act_4cams': {
        'dataset_dir': DATA_DIR + '/alarm_random_pos',
        'episode_len': 1000,  # 900
        'camera_names': ['cam_high',
                         'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         ]
    },

    'stapler_simple_act_4cams': {
        'dataset_dir': DATA_DIR + '/stapler_simple',
        'episode_len': 600,
        'camera_names': ['cam_high',
                         'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         ]
    },
    "aloha_test": {
        "dataset_dir": DATA_DIR + "/aloha_test",
        "num_episodes": 50,
        "episode_len": 500,
        "camera_names": ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"],
    },


    # 'weighing_tie1': {  # include loss
    #     'dataset_dir': [
    #         DATA_DIR + '/weighing/open_balance',
    #         DATA_DIR + '/weighing/place_battery',
    #     ],  # only the first dataset_dir is used for val
    #     'stats_dir': [
    #         DATA_DIR + '/weighing/open_balance',
    #         DATA_DIR + '/weighing/place_battery',
    #     ],
    #     'train_ratio': 0.99,  # ratio of train data from the first dataset_dir
    #     'episode_len': 1800,
    #     'camera_names': ['cam_high',
    #                      'cam_low',
    #                      'cam_left_wrist',
    #                      'cam_right_wrist',
    #                      ]
    # },

    'weighing_tie1': {  # include loss
        'dataset_dir': [
            DATA_DIR + '/weighing/place_battery',
            DATA_DIR + '/weighing/remove_battery',
        ],  # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/weighing/place_battery',
            DATA_DIR + '/weighing/remove_battery',
        ],
        'train_ratio': 0.99,  # ratio of train data from the first dataset_dir
        'episode_len': 1000,
        'camera_names': ['cam_high',
                         'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         ]
    },

    # 'weighing_tie2': {  # include loss
    #     'dataset_dir': [
    #         DATA_DIR + '/weighing/open_balance',
    #         DATA_DIR + '/weighing/place_red_cube',
    #         DATA_DIR + '/weighing/shutdown_balance',
    #     ],  # only the first dataset_dir is used for val
    #     'stats_dir': [
    #         DATA_DIR + '/weighing/open_balance',
    #         DATA_DIR + '/weighing/place_red_cube',
    #         DATA_DIR + '/weighing/shutdown_balance',
    #     ],
    #     'train_ratio': 0.99,  # ratio of train data from the first dataset_dir
    #     'episode_len': 1500,
    #     'camera_names': ['cam_high',
    #                      'cam_low',
    #                      'cam_left_wrist',
    #                      'cam_right_wrist',
    #                      ]
    # },

'weighing_tie2': {  # include loss
        'dataset_dir': [
            DATA_DIR + '/weighing/place_red_cube',
            DATA_DIR + '/weighing/remove_red_cube',
            DATA_DIR + '/weighing/shutdown_balance',
        ],  # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/weighing/place_red_cube',
            DATA_DIR + '/weighing/remove_red_cube',
            DATA_DIR + '/weighing/shutdown_balance',
        ],
        'train_ratio': 0.99,  # ratio of train data from the first dataset_dir
        'episode_len': 1000,
        'camera_names': ['cam_high',
                         'cam_low',
                         'cam_left_wrist',
                         'cam_right_wrist',
                         ]
    },

    'cup_load_data_collection': {
            'dataset_dir': DATA_DIR + '/cup_load',
            'episode_len': 1000,  # 900
            'camera_names': ['cam_high',
                             'cam_low',
                             'cam_left_wrist',
                             'cam_right_wrist',
                             ]
        },
    
    'apple_grasping': {
        'dataset_dir': DATA_DIR + '/apple_grasping',
        'episode_len': 450,  # e.g., 15 seconds * 30FPS = 450
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },

    'aloha_wear_shoe': {
        'dataset_dir': DATA_DIR + '/aloha_wear_shoe',
        'num_episodes': 50,
        'episode_len': 600,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_grasp_apple': {  # test task only
        'dataset_dir': DATA_DIR + '/aloha_grasp_apple',
        'num_episodes': 85,
        'episode_len': 600,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },

    # ── 正式实验任务 ──
    'aloha_flip_switch': {  # Level 1: visual-dominant
        'dataset_dir': DATA_DIR + '/aloha_flip_switch',
        'num_episodes': 50,
        'episode_len': 300,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_hidden_property_grasp': {  # Level 2: hidden-property force-critical
        'dataset_dir': DATA_DIR + '/aloha_hidden_property_grasp',
        'num_episodes': 50,
        'episode_len': 450,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_slip_hold_or_pull': {  # Level 2/3: controlled slip / grasp stability
        'dataset_dir': DATA_DIR + '/aloha_slip_hold_or_pull',
        'num_episodes': 50,
        'episode_len': 600,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_pressure_wipe': {  # Level 2/3: simplified continuous-contact force control
        'dataset_dir': DATA_DIR + '/aloha_pressure_wipe',
        'num_episodes': 50,
        'episode_len': 450,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_peg_insertion': {  # Level 3: fine-tactile slip detection
        'dataset_dir': DATA_DIR + '/aloha_peg_insertion',
        'num_episodes': 50,
        'episode_len': 600,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_towel_unfold': {  # Extension: high-variance deformable manipulation
        'dataset_dir': DATA_DIR + '/aloha_towel_unfold',
        'num_episodes': 50,
        'episode_len': 600,
        'camera_names': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
    },

}


### ALOHA fixed constants
DT = 1/30  # FPS=30
FPS = 30

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]
#START_ARM_POSE = [0, -0.8, 0.8, 0, 0, 0, 0.02239, -0.02239,  0, -0.8, 0.8, 0, 0, 0, 0.02239, -0.02239]

# Sleep positions for robots (6 joints: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate)
PUPPET_SLEEP_POSITION = (0, -1.7, 1.55, 0.12, 0.65, 0)
MASTER_SLEEP_POSITION = (0, -1.76, 1.55, 0, 0.0, 0)

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.7409
MASTER_GRIPPER_JOINT_CLOSE = -0.0614
PUPPET_GRIPPER_JOINT_OPEN = 0.0414
PUPPET_GRIPPER_JOINT_CLOSE = -0.9265  # @gnq -0.6213 -> 0

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2



SIM_TASK_CONFIGS = {
    'sim_transfer_cube_scripted':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    'sim_transfer_cube_human':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_human',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_scripted': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_human': {
        'dataset_dir': DATA_DIR + '/sim_insertion_human',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['top']
    },
    'all': {
        'dataset_dir': DATA_DIR + '/',
        'num_episodes': None,
        'episode_len': None,
        'name_filter': lambda n: 'sim' not in n,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },

    'sim_transfer_cube_scripted_mirror':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted_mirror',
        'num_episodes': None,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_scripted_mirror': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted_mirror',
        'num_episodes': None,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

}
