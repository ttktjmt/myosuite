import mujoco

model = mujoco.MjModel.from_xml_path("myoarm_bionic_bimanual.xml") # adjust the path to your model
mujoco.mj_saveModel(model, "myoarm_bionic_bimanual.mjb", None)

print("Model successfully saved!")