import pinocchio as pin
import numpy as np
import xml.etree.ElementTree as ET

def urdf_movable_joint_names(path: str) -> list[str]:
    root = ET.parse(path).getroot()
    names = []
    for j in root.findall("joint"):
        if j.get("type") != "fixed":
            name = j.get("name")
            if name:
                names.append(name)
    return names

def all_joint_positions(model: pin.Model, q: np.ndarray):
    """
    Returns dict: joint_name -> position (float or np.ndarray for multi-DoF joints)
    Includes:
      - normal joints from q
      - mimic joints computed as s*q_ref + o
    """
    out = {}

    # Normal joints (skip joint 0 = "universe")
    for jid in range(1, model.njoints):
        j = model.joints[jid]
        name = model.names[jid]

        # Mimic joints have nq=0, so skip here (we'll compute below)
        if j.nq == 0:
            continue

        qj = q[j.idx_q : j.idx_q + j.nq]
        out[name] = float(qj) if qj.size == 1 else qj.copy()

    # Mimic joints: q_mim = s*q_ref + o
    for mim_jid, ref_jid in zip(model.mimicking_joints, model.mimicked_joints):
        mim_name = model.names[mim_jid]
        ref = model.joints[ref_jid]

        jm = model.joints[mim_jid].extract()   # JointModelMimic
        s = jm.scaling
        o = jm.offset

        q_ref = q[ref.idx_q : ref.idx_q + ref.nq]
        q_mim = s * q_ref + o
        out[mim_name] = float(q_mim) if q_mim.size == 1 else q_mim.copy()

    return out


def set_q_from_joint_dict(model: pin.Model, q: np.ndarray, joint_dict: dict[str, float]) -> np.ndarray:
    for name, val in joint_dict.items():
        jid = model.getJointId(name)
        if jid == model.njoints:
            continue  # not in model
        idx_q = model.idx_qs[jid]
        nq = model.nqs[jid]
        if nq == 1:
            q[idx_q] = val
    return q
