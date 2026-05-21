from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

TEMPLATE_ASSETS_DATA_DIR = Path(__file__).resolve().parent

##
# Configuration
##

SO_ARM101_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=True,
        replace_cylinders_with_capsules=True,
        asset_path=f"{TEMPLATE_ASSETS_DATA_DIR}/urdf/so_arm101.urdf",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "shoulder_pan":  0.0,
            "shoulder_lift": 0.0,
            "elbow_flex":    0.0,
            "wrist_flex":    1.57,
            "wrist_roll":    0.0,
            "gripper":       0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Shoulder Pan      moves: ALL masses                (~0.8kg total)
        # Shoulder Lift     moves: Everything except base    (~0.65kg)
        # Elbow             moves: Lower arm, wrist, gripper (~0.38kg)
        # Wrist Pitch       moves: Wrist and gripper         (~0.24kg)
        # Wrist Roll        moves: Gripper assembly          (~0.14kg)
        # Jaw               moves: Only moving jaw           (~0.034kg)
        "arm": DCMotorCfg(
            joint_names_expr=["shoulder_.*", "elbow_flex", "wrist_.*"],
            effort_limit_sim=1.47,
            velocity_limit_sim=3.0,
            velocity_limit=3.0,
            saturation_effort=1.47,
            stiffness={
                "shoulder_pan":  16.0,
                "shoulder_lift": 16.0,
                "elbow_flex":    16.0,
                "wrist_flex":    16.0,
                "wrist_roll":    16.0,
            },
            damping={
                "shoulder_pan":  3.2,
                "shoulder_lift": 3.2,
                "elbow_flex":    3.2,
                "wrist_flex":    3.2,
                "wrist_roll":    3.2,
            },
        ),
        "gripper": DCMotorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=1.47,
            velocity_limit_sim=3.0,
            velocity_limit=3.0,
            saturation_effort=1.47,
            stiffness=16.0,
            damping=3.2,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)