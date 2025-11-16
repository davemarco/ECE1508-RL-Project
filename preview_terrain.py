import os
# 1) Make sure we use EGL (headless). Do this BEFORE importing mujoco.
os.environ.setdefault("MUJOCO_GL", "egl")
# Optional: pick a GPU if multiple
# os.environ.setdefault("EGL_DEVICE_ID", "0")

import numpy as np
import mediapy as media
import mujoco

XML = "humanoid_uneven.xml"  # update if needed
H, W = 480, 640
STEPS = 1500
CAMERA_NAME = "side"  # or None to use the free camera

# Load model/data
model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)

# Try to find a camera id (OK if missing)
try:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME) if CAMERA_NAME else -1
except Exception:
    cam_id = -1

frames = []
# 2) Use the context manager so resources always release (no _mjr_context errors)
with mujoco.Renderer(model, height=H, width=W) as renderer:
    for _ in range(STEPS):
        mujoco.mj_step(model, data)
        if cam_id >= 0:
            renderer.update_scene(data, camera=cam_id)
        else:
            renderer.update_scene(data)
        frames.append(renderer.render())

# Write video
fps = int(1.0 / model.opt.timestep / 4)  # downsample a bit
media.write_video("rollout.mp4", frames, fps=fps)
print("Saved rollout.mp4")
