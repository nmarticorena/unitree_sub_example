from vuer import Vuer
import asyncio
import time
from vuer.schemas import DefaultScene, Urdf, OrbitControls
from robot_control import G1_29_ArmController
from robot_hand_inspire import Inspire_Controller_DFX
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
import urllib.request
import numpy as np
from multiprocessing import Process, Array
import xml.etree.ElementTree as ET
ChannelFactoryInitialize(1)

robot_control = G1_29_ArmController(motion_mode = True,simulation_mode = True)


robot_hand = Inspire_Controller_DFX()

def urdf_movable_joint_names(url: str) -> list[str]:
    with urllib.request.urlopen(url) as resp:
        urdf_xml = resp.read()  # bytes
    root = ET.fromstring(urdf_xml)
    names = []
    for j in root.findall("joint"):
        if j.get("type") != "fixed":
            name = j.get("name")
            if name:
                names.append(name)
    return names


app = Vuer()
URDF_PATH = "https://raw.githubusercontent.com/unitreerobotics/unitree_ros/refs/heads/master/robots/g1_description/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"
print(urdf_movable_joint_names(URDF_PATH))
@app.spawn(start=True)
async def main(sess):
    # print(robot_control.get_current_motor_q())
    urdf = Urdf(
        src=URDF_PATH,
    )
    sess.set @ DefaultScene(
    #     urdf,
        bgChildren=[OrbitControls(key="OrbitControls")],
        up=[0, 0, 1],
    )
    
    
    while True:
        
        robotq = np.concatenate((robot_control.get_current_motor_q()[:29], robot_hand.get_state()))
        print(robot_hand.get_state())
        jointvals = {name: val.item() for name, val in zip(urdf_movable_joint_names(URDF_PATH), robotq)}
        sess.upsert @ Urdf(
            src=URDF_PATH,
            jointValues=jointvals,
            key = "robot"
        )
        # print(robot_hand.get_current_motor_q())

        await asyncio.sleep(0.1)
