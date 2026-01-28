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

class Inspire_Controller_FTP:
    def __init__(self, fps = 100.0, Unit_Test = False, simulation_mode = False):
        logger_mp.info("Initialize Inspire_Controller_FTP...")
        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode

        # Initialize hand state subscribers
        self.LeftHandState_subscriber = ChannelSubscriber(kTopicInspireFTPLeftState, inspire_hand_state)
        self.LeftHandState_subscriber.Init() # Consider using callback if preferred: Init(callback_func, period_ms)
        self.RightHandState_subscriber = ChannelSubscriber(kTopicInspireFTPRightState, inspire_hand_state)
        self.RightHandState_subscriber.Init()

        # Shared Arrays for hand states ([0,1] normalized values)
        self.left_hand_state_array  = Array('d', Inspire_Num_Motors, lock=True)
        self.right_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)

        # Initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        # Wait for initial DDS messages (optional, but good for ensuring connection)
        wait_count = 0
        while not (any(self.left_hand_state_array) or any(self.right_hand_state_array)):
            if wait_count % 100 == 0: # Print every second
                logger_mp.info(f"[Inspire_Controller_FTP] Waiting to subscribe to hand states from DDS (L: {any(self.left_hand_state_array)}, R: {any(self.right_hand_state_array)})...")
            time.sleep(0.01)
            wait_count += 1
            if wait_count > 500: # Timeout after 5 seconds
                logger_mp.warning("[Inspire_Controller_FTP] Warning: Timeout waiting for initial hand states. Proceeding anyway.")
                break
        logger_mp.info("[Inspire_Controller_FTP] Initial hand states received or timeout.")

        

    def _subscribe_hand_state(self):
        logger_mp.info("[Inspire_Controller_FTP] Subscribe thread started.")
        while True:
            # Left Hand
            left_state_msg = self.LeftHandState_subscriber.Read()
            if left_state_msg is not None:
                if hasattr(left_state_msg, 'angle_act') and len(left_state_msg.angle_act) == Inspire_Num_Motors:
                    with self.left_hand_state_array.get_lock():
                        for i in range(Inspire_Num_Motors):
                            self.left_hand_state_array[i] = left_state_msg.angle_act[i] / 1000.0
                else:
                    logger_mp.warning(f"[Inspire_Controller_FTP] Received left_state_msg but attributes are missing or incorrect. Type: {type(left_state_msg)}, Content: {str(left_state_msg)[:100]}")
            # Right Hand
            right_state_msg = self.RightHandState_subscriber.Read()
            if right_state_msg is not None:
                if hasattr(right_state_msg, 'angle_act') and len(right_state_msg.angle_act) == Inspire_Num_Motors:
                    with self.right_hand_state_array.get_lock():
                        for i in range(Inspire_Num_Motors):
                            self.right_hand_state_array[i] = right_state_msg.angle_act[i] / 1000.0
                else:
                    logger_mp.warning(f"[Inspire_Controller_FTP] Received right_state_msg but attributes are missing or incorrect. Type: {type(right_state_msg)}, Content: {str(right_state_msg)[:100]}")

            time.sleep(0.002)
            
    def get_state(self):
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
                right_[idx] = normalize(right[idx], 0.0, 0.5)
            elif idx == 5:
                left[idx]  = normalize(left[idx], -0.1, 1.3)
                right[idx] = normalize(right[idx], -0.1, 1.3)

        left_mimic = [x*1000/2 for x in left for _ in range(2)]
        
        right_mimic = [x*1000/2 for x in right for _ in range(2)]
        
        return np.array((left_mimic, right_mimic)).flatten()
        
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

        hand_msg = self.HandState_subscriber.Read() 
        breakpoint()
        
        
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
            
    def convert_mimic(self, motor_pos):
        # motor_pos *= -1
        # This is quite nasty
        # motor_pos [6] order:
            # Pinky, Ring, Middle, Index, Thumb bend, Thumb rotation
        # angles [12]
            # L_thumb_proximal_yaw_joint', 'L_thumb_proximal_pitch_joint', 'L_thumb_intermediate_joint', 'L_thumb_distal_joint', 'L_index_proximal_joint', 'L_index_intermediate_joint', 'L_middle_proximal_joint', 'L_middle_intermediate_joint', 'L_ring_proximal_joint', 'L_ring_intermediate_joint', 'L_pinky_proximal_joint', 'L_pinky_intermediate_joint'
        angles = np.zeros(12)
        # Thumb proximal yaw
        # L_thumb_proximal_yaw_joint'
        angles[0] = motor_pos[5]
        # L_thumb_proximal_pitch_joint'
        angles[1] = motor_pos[4]
        # 'L_thumb_intermediate_joint'
        angles[2] = angles[1] * 1.6
        # 'L_thumb_distal_joint'
        angles[3] = angles[1] * 2.4
        
        # 'L_index_proximal_joint'
        angles[4] = motor_pos[3]
        # 'L_index_intermediate_joint'
        angles[5] = angles[4] * 1
        
        # 'L_middle_proximal_joint'
        angles[6] = motor_pos[2]
        # 'L_middle_intermediate_joint'
        angles[7] = angles[6] * 1
        
        
        # 'L_ring_proximal_joint'
        angles[8] = motor_pos[1]
        # 'L_ring_intermediate_joint'
        angles[9] = angles[8] * 1
        
        
        # 'L_pinky_proximal_joint'
        angles[10] = motor_pos[0]
        # 'L_pinky_intermediate_joint'
        angles[11] = angles[10] * 1
        
        
        
        
        return angles
        
    
    def get_state(self):
        
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
        
        left_angle = self.convert_mimic(np.array(left))
        right_angle = self.convert_mimic(np.array(right))
            
        return np.array((left_angle, right_angle)).flatten()
            
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