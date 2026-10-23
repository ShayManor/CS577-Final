import glob
import os

import habitat_sim
import numpy as np

HM3D = "data/scene_datasets/hm3d"
CONFIG = f"{HM3D}/hm3d_annotated_val_basis.scene_dataset_config.json"
H, W = 240, 320


def first_annotated_scene():
    txt = sorted(glob.glob(f"{HM3D}/0*/*.semantic.txt"))[0]
    d = os.path.basename(os.path.dirname(txt))
    return d, f"{HM3D}/{d}/{d.split('-', 1)[1]}.basis.glb"


def test_load_scene_and_render():
    _, glb = first_annotated_scene()
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = glb
    backend.scene_dataset_config_file = CONFIG
    backend.gpu_device_id = 0
    cam = habitat_sim.CameraSensorSpec()
    cam.uuid = "rgb"
    cam.sensor_type = habitat_sim.SensorType.COLOR
    cam.resolution = [H, W]
    cam.position = [0.0, 1.5, 0.0]
    agent = habitat_sim.agent.AgentConfiguration(sensor_specifications=[cam])
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent]))
    try:
        assert len(sim.semantic_scene.objects) > 0
        assert sim.pathfinder.is_loaded
        pos = sim.pathfinder.get_random_navigable_point()
        state = habitat_sim.AgentState()
        state.position = pos
        sim.get_agent(0).set_state(state)
        rgb = sim.get_sensor_observations()["rgb"]
        assert rgb.shape == (H, W, 4)
        assert np.var(rgb.astype(float)) > 0
    finally:
        sim.close()
