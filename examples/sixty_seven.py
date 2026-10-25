import mujoco
import mujoco.viewer
import numpy as np
import time

# 1. Load the robot
model = mujoco.MjModel.from_xml_path(
    "mujoco/models/bimanual.xml"
)
data = mujoco.MjData(model)

# 2. Inspect the actuators
print("Number of actuators:", model.nu)
for i in range(model.nu):
    print(i, model.actuator(i).name)

# 67 setup (runs once)
R_ELBOW = model.actuator("right_rev4_ctrl").id
L_ELBOW = model.actuator("left_rev4_ctrl").id
R_WRIST = model.actuator("right_rev7_ctrl").id
L_WRIST = model.actuator("left_rev7_ctrl").id
ELBOW_BASE = 1.57
WRIST_BASE = 0
A = 0.3
W = 6.0

# targets for joints 1–7 (start at 0, then adjust)
R_POSE = [0, 0, 0, ELBOW_BASE, 0, 0, WRIST_BASE]
L_POSE = [0, 0, 0, ELBOW_BASE, 0, 0, WRIST_BASE]

# calculates how hard to push a joint so it moves to a target angle and stays there
def pd(joint, target, kp=20.0, kd=1.0):
    jid = model.joint(joint).id
    q  = data.qpos[model.jnt_qposadr[jid]]
    dq = data.qvel[model.jnt_dofadr[jid]]
    return kp * (target - q) - kd * dq


# 3. Start the simulator
with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        start = time.time()

        # # 4. Apply a changing motor command:
        # #set all 16 DOF actuators to 0
        # data.ctrl[:] = 0
        
        # ### testing code
        # k = int(data.time // 3) % model.nu     # switch actuator every 3 s
        # data.ctrl[:] = 0
        # data.ctrl[k] = 0.5 * np.sin(2 * data.time)
        # if int(data.time * 100) % 300 == 0:    # print when it switches
        #     print("moving:", k, model.actuator(k).name)

        # 4. 67 motion (replaces the old step 4)
        data.ctrl[:] = 0
        bob = A * np.sin(W * data.time)
        for j in range(1, 8):
            r_t = R_POSE[j-1] + (bob if j == 4 else 0)
            l_t = L_POSE[j-1] - (bob if j == 4 else 0)
            data.ctrl[model.actuator(f"right_rev{j}_ctrl").id] = pd(f"right_rev{j}", r_t)
            data.ctrl[model.actuator(f"left_rev{j}_ctrl").id]  = pd(f"left_rev{j}",  l_t)

        ### wrist testing
        for name in ["right_rev5", "right_rev6"]:
            q = data.qpos[model.jnt_qposadr[model.joint(name).id]]
            c = data.ctrl[model.actuator(name + "_ctrl").id]
            print(name, "cmd", round(c, 2), "actual", round(q, 2))

        # 5. Advance the physics simulation
        mujoco.mj_step(model, data)

        # 6. Update the screen
        viewer.sync()

        # 7. Keep the simulation near real-time
        elapsed = time.time() - start
        time.sleep(max(0, model.opt.timestep - elapsed))
