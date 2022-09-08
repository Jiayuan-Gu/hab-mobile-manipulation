import json

from habitat_sim import Simulator
from habitat_sim.gfx import LightInfo, LightPositionModel
from habitat_sim.nav import NavMeshSettings


def load_light_setup(json_filepath):
    """Load light setup from a json file."""
    with open(json_filepath) as json_file:
        config = json.load(json_file)
        light_setup = []
        for light in config["lights"].values():
            position = [float(x) for x in light["position"][0:3]]
            w = 1.0  # 1 for point light and 0 for directional light
            color_scale = float(light["intensity"])
            color = [float(c * color_scale) for c in light["color"]]
            light_setup.append(
                LightInfo(
                    vector=position + [w],
                    color=color,
                    model=LightPositionModel.Global,
                )
            )
    return light_setup


def get_navmesh_settings(agent_config):
    navmesh_settings = NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_radius = agent_config.RADIUS
    navmesh_settings.agent_height = agent_config.HEIGHT
    navmesh_settings.agent_max_climb = 0.05
    return navmesh_settings


def get_object_handle_by_id(sim: Simulator, object_id: int):
    rigid_obj_mgr = sim.get_rigid_object_manager()
    art_obj_mgr = sim.get_articulated_object_manager()

    if rigid_obj_mgr.get_library_has_id(object_id):
        obj = rigid_obj_mgr.get_object_by_id(object_id)
    elif art_obj_mgr.get_library_has_id(object_id):
        obj = art_obj_mgr.get_object_by_id(object_id)
    else:  # static scene
        assert object_id == -1, object_id
        return None

    return obj.handle
