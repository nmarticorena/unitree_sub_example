from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_                           # idl
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
import numpy as np
from enum import IntEnum
import threading
import time
from multiprocessing import Process, Array
from dataclasses import dataclass
import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types

Inspire_Num_Motors = 6

@dataclass
@annotate.final
@annotate.autoid("sequential")
class inspire_hand_state(idl.IdlStruct, typename="inspire.inspire_hand_state"):
    pos_act: types.sequence[types.int16, 6]
    angle_act: types.sequence[types.int16, 6]
    force_act: types.sequence[types.int16, 6]
    current: types.sequence[types.int16, 6]
    err: types.sequence[types.uint8, 6]
    status: types.sequence[types.uint8, 6]
    temperature: types.sequence[types.uint8, 6]

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

kTopicInspireFTPLeftCommand   = "rt/inspire_hand/ctrl/l"
kTopicInspireFTPRightCommand  = "rt/inspire_hand/ctrl/r"
kTopicInspireFTPLeftState  = "rt/inspire_hand/state/l"
kTopicInspireFTPRightState = "rt/inspire_hand/state/r"

kTopicInspireDFXCommand = "rt/inspire/cmd"
kTopicInspireDFXState = "rt/inspire/state"


URDF_JOINT_MAP = {
    # Right
    "kRightHandPinky": "R_pinky_proximal_joint",
    "kRightHandRing": "R_ring_proximal_joint",
    "kRightHandMiddle": "R_middle_proximal_joint",
    "kRightHandIndex": "R_index_proximal_joint",
    "kRightHandThumbBend": "R_thumb_proximal_pitch_joint",
    "kRightHandThumbRotation": "R_thumb_proximal_yaw_joint",


    "kLeftHandPinky": "L_pinky_proximal_joint",
    "kLeftHandRing": "L_ring_proximal_joint",
    "kLeftHandMiddle": "L_middle_proximal_joint",
    "kLeftHandIndex": "L_index_proximal_joint",
    "kLeftHandThumbBend": "L_thumb_proximal_pitch_joint",
    "kLeftHandThumbRotation": "L_thumb_proximal_yaw_joint",

    }

class Inspire_Controller_DFX:
    def __init__(self, fps = 100.0, Unit_Test = False, simulation_mode = False):
        logger_mp.info("Initialize Inspire_Controller_DFX...")
        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode


        self.HandState_subscriber = ChannelSubscriber(kTopicInspireDFXState, MotorStates_)
        self.HandState_subscriber.Init()

        # Shared Arrays for hand states
        self.left_hand_state_array  = Array('d', Inspire_Num_Motors, lock=True)
        self.right_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

    def _subscribe_hand_state(self):
        while True:
            hand_msg  = self.HandState_subscriber.Read()
            if hand_msg is not None:
                for idx, id in enumerate(Inspire_Left_Hand_JointIndex):
                    self.left_hand_state_array[idx] = hand_msg.states[id].q
                for idx, id in enumerate(Inspire_Right_Hand_JointIndex):
                    self.right_hand_state_array[idx] = hand_msg.states[id].q
            time.sleep(0.002)

    def get_state(self):
        out = {}
        def put(enum_item, value):
            ctrl_name = enum_item.name

            key = URDF_JOINT_MAP.get(ctrl_name, ctrl_name)  # fallback
            out[key] = float(value)

        left = np.array(self.left_hand_state_array)
        right = np.array(self.right_hand_state_array)

        def normalize(val, min_val, max_val):
            return np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)

        for idx in range(Inspire_Num_Motors):
            if idx <= 3:
                left[idx]  = normalize(left[idx], 0.0, 1.7)
                right[idx] = normalize(right[idx], 0.0, 1.7)
            elif idx == 4:
                left[idx]  = normalize(left[idx], 0.0, 0.5)
                right[idx] = normalize(right[idx], 0.0, 0.5)
            elif idx == 5:
                left[idx]  = normalize(left[idx], -0.1, 1.3)
                right[idx] = normalize(right[idx], -0.1, 1.3)
            put(Inspire_Right_Hand_JointIndex(idx), right[idx])
            put(Inspire_Left_Hand_JointIndex(idx+6), left[idx])

        return out

# Update hand state, according to the official documentation:
# 1. https://support.unitree.com/home/en/G1_developer/inspire_dfx_dexterous_hand
# 2. https://support.unitree.com/home/en/G1_developer/inspire_ftp_dexterity_hand
# the state sequence is as shown in the table below
# ┌──────┬───────┬──────┬────────┬────────┬────────────┬────────────────┬───────┬──────┬────────┬────────┬────────────┬────────────────┐
# │ Id   │   0   │  1   │   2    │   3    │     4      │       5        │   6   │  7   │   8    │   9    │    10      │       11       │
# ├──────┼───────┼──────┼────────┼────────┼────────────┼────────────────┼───────┼──────┼────────┼────────┼────────────┼────────────────┤
# │      │                    Right Hand                                │                   Left Hand                                  │
# │Joint │ pinky │ ring │ middle │ index  │ thumb-bend │ thumb-rotation │ pinky │ ring │ middle │ index  │ thumb-bend │ thumb-rotation │
# └──────┴───────┴──────┴────────┴────────┴────────────┴────────────────┴───────┴──────┴────────┴────────┴────────────┴────────────────┘
class Inspire_Right_Hand_JointIndex(IntEnum):
    kRightHandPinky = 0
    kRightHandRing = 1
    kRightHandMiddle = 2
    kRightHandIndex = 3
    kRightHandThumbBend = 4
    kRightHandThumbRotation = 5

class Inspire_Left_Hand_JointIndex(IntEnum):
    kLeftHandPinky = 6
    kLeftHandRing = 7
    kLeftHandMiddle = 8
    kLeftHandIndex = 9
    kLeftHandThumbBend = 10
    kLeftHandThumbRotation = 11
