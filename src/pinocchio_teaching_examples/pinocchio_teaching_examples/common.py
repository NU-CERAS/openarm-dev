"""Load the same robot used by RViz, without starting any ROS nodes."""
from pathlib import Path
import numpy as np
import pinocchio as pin
import xacro
from ament_index_python.packages import get_package_share_directory


def load_arm():
    share = Path(get_package_share_directory('teaching_arm_description'))
    xml = xacro.process_file(str(share / 'urdf/two_link_arm.urdf.xacro')).toxml()
    model = pin.buildModelFromXML(xml)
    return model, model.createData(), model.getFrameId('tool0')


def hand_position(model, data, frame_id, q):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[frame_id].translation.copy()


def analytical_position(q, length1=0.6, length2=0.4):
    # Rotation about +Y sends +X toward -Z. No Y displacement in this planar arm.
    shoulder, elbow = q
    return np.array([
        length1 * np.cos(shoulder) + length2 * np.cos(shoulder + elbow),
        0.0,
        -length1 * np.sin(shoulder) - length2 * np.sin(shoulder + elbow),
    ])
