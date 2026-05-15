from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

TEMPLATE_ASSETS_DATA_DIR = Path(__file__).resolve().parent

##
# Configuration
##

RL_BOWL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Bowl",  # one bowl per env, matches Isaac Lab env namespacing
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{TEMPLATE_ASSETS_DATA_DIR}/urdf/rl_bowl.urdf",
        fix_base=True,
        make_instanceable=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0,
                damping=0,
            ),
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=False,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,
            max_depenetration_velocity=1.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        # Z offset of +0.003 m lifts the mesh so the base lip sits flush on the surface.
        # The STL has Z_min = -3 mm, so this zeroes it out.
        # Adjust pos to wherever the bowl should sit in your scene (e.g. on the table).
        pos=(0.5, 0.0, 0.003),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)
