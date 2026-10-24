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

# 3. Start the simulator
with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        start = time.time()

        # 4. Apply a changing motor command:
        #set all 16 DOF actuators to 0
        data.ctrl[:] = 0
        #sinusoidal force on 1st joint of right arm
        data.ctrl[0] = 2.0 * np.sin(1*data.time) 

        # 5. Advance the physics simulation
        mujoco.mj_step(model, data)

        # 6. Update the screen
        viewer.sync()

        # 7. Keep the simulation near real-time
        elapsed = time.time() - start
        time.sleep(max(0, model.opt.timestep - elapsed))
