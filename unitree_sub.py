from vuer import Vuer
import asyncio
import time
from vuer.schemas import DefaultScene, Urdf, OrbitControls
from robot_control import G1_29_ArmController
from robot_hand_inspire import Inspire_Controller_DFX
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
import numpy as np
import utils
import pinocchio
import pathlib

from utils import urdf_movable_joint_names, all_joint_positions

URDF_PATH = "g1.urdf"
URDF_URL = "https://raw.githubusercontent.com/unitreerobotics/unitree_ros/refs/heads/master/robots/g1_description/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"


ChannelFactoryInitialize(1)

robot_control = G1_29_ArmController(motion_mode = True,simulation_mode = True)
robot_hand = Inspire_Controller_DFX()
robot_model = pinocchio.buildModelFromUrdf(URDF_PATH, mimic = True)

key_path = pathlib.Path(__file__).parent / "key.pem"
cert_path = pathlib.Path(__file__).parent / "cert.pem"
app = Vuer(host = "0.0.0.0", port = 8012)
app.cert = cert_path
app.key = key_path

@app.spawn(start=True)
async def main(sess):
    sess.set @ DefaultScene(
        bgChildren=[OrbitControls(key="OrbitControls")],
        up=[0, 1, 0],
    )


    while True:

        robot_state = robot_control.get_motor_states()
        hand_state = robot_hand.get_state()

        state = {**robot_state, **hand_state}

        q = pinocchio.neutral(robot_model)
        q =utils.set_q_from_joint_dict(robot_model, q,  state)

        jointvals = all_joint_positions(robot_model, q)


        sess.upsert @ Urdf(
            src=URDF_URL,
            jointValues=jointvals,
            key = "robot",
            position=[0.0, 1.5, -1.2],   # move robot instead of VR camera (tune axes)
        )

        await asyncio.sleep(0.1)
