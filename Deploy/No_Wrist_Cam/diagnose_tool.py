# -*- coding: utf-8 -*-
"""
diagnose_deployment.py
======================
Diagnose script - does NOT move the robot.
Reads current robot state once and prints all critical values:
  - Raw encoder values
  - Observation vector (all 27 dimensions)
  - Policy output (raw actions)
  - Computed arm targets
  - EE position via FK

Usage (same args as deploy script):
  python diagnose_deployment.py ^
      --checkpoint Deploy/Policy/only_init_cube_obs/exported/policy.pt ^
      --urdf_path Deploy/No_Wrist_Cam/so_arm101.urdf ^
      --robot_port COM5 ^
      --object_pos 0.23 0.09 0.00 ^
      --bowl_pos 0.43 0.00 0.00

Without robot (--mock):
  python diagnose_deployment.py --checkpoint ... --urdf_path ... --mock
"""

import argparse
import sys
import numpy as np
import torch

# -- Konstanten (identisch zu deploy_script) ----------------------------------

ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT_NAME = "gripper"
ALL_JOINT_NAMES = ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME]
NUM_ARM_JOINTS = 5
NUM_ACTIONS = 6
NUM_OBS = 27

DEFAULT_JOINT_POS_RAD = np.array([0.00, -0.40, -0.30, 1.57, -1.57, 0.20], dtype=np.float64)

GRIPPER_OPEN_CMD_RAD  =  0.5
GRIPPER_CLOSE_CMD_RAD = -0.1
GRIPPER_RAD_MIN = GRIPPER_CLOSE_CMD_RAD
GRIPPER_RAD_MAX = GRIPPER_OPEN_CMD_RAD
ARM_ACTION_SCALE     = 0.5
GRIPPER_ACTION_SCALE = 0.3
BOWL_HOVER_HEIGHT    = 0.12
EE_OFFSET = np.array([0.01, 0.0, -0.09])

SEP = "-" * 60


def build_fk_model(urdf_path):
    import pinocchio as pin
    model = pin.buildModelFromUrdf(urdf_path)
    data  = model.createData()
    def _jidx(name):
        return model.joints[model.getJointId(name)].idx_q
    j_idx = {n: _jidx(n) for n in ALL_JOINT_NAMES}
    frame_id = model.getFrameId("gripper_link")
    return model, data, j_idx, frame_id


def get_ee_pos(q_abs, pin_model, pin_data, j_idx, gripper_frame_id):
    import pinocchio as pin
    q = np.zeros(pin_model.nq)
    for name, idx in j_idx.items():
        q[idx] = q_abs[name]
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    T = pin_data.oMf[gripper_frame_id]
    return (T.translation + T.rotation @ EE_OFFSET).astype(np.float32)


def gripper_pct_to_rad(pct):
    return GRIPPER_RAD_MIN + (pct / 100.0) * (GRIPPER_RAD_MAX - GRIPPER_RAD_MIN)


def gripper_rad_to_pct(rad):
    return float(np.clip((rad - GRIPPER_RAD_MIN) / (GRIPPER_RAD_MAX - GRIPPER_RAD_MIN) * 100.0, 0.0, 100.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--urdf_path", required=True)
    parser.add_argument("--robot_port", default="COM5")
    parser.add_argument("--robot_id", default="follower_arm_v2")
    parser.add_argument("--robot_type", default="so101_follower")
    parser.add_argument("--object_pos", nargs=3, type=float, default=[0.30, 0.00, 0.01])
    parser.add_argument("--bowl_pos",   nargs=3, type=float, default=[0.43, 0.00, 0.00])
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    object_pos = np.array(args.object_pos, dtype=np.float32)
    bowl_pos   = np.array(args.bowl_pos,   dtype=np.float32)

    # -- 1. FK-Modell --------------------------------------------------------
    print(f"\n{SEP}")
    print("1. PINOCCHIO FK-MODELL")
    print(SEP)
    pin_model, pin_data, j_idx, gripper_frame_id = build_fk_model(args.urdf_path)
    print(f"  nq={pin_model.nq}, frames={pin_model.nframes}")
    print(f"  gripper_link frame_id={gripper_frame_id}")
    print(f"  Joint-Index-Map: {j_idx}")

    # -- 2. Roboter-State lesen -----------------------------------------------
    print(f"\n{SEP}")
    print("2. ROBOTER-STATE")
    print(SEP)

    if args.mock:
        print("  [MOCK] Verwende Default-Positionen als aktuellen State.")
        raw_obs = {f"{m}.pos": float(np.rad2deg(DEFAULT_JOINT_POS_RAD[i]))
                   for i, m in enumerate(ARM_JOINT_NAMES)}
        raw_obs["gripper.pos"] = gripper_rad_to_pct(DEFAULT_JOINT_POS_RAD[5])
    else:
        from lerobot.robots.so_follower import SO100Follower, SO101Follower
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        robot_cls = SO101Follower if args.robot_type == "so101_follower" else SO100Follower
        robot_cfg = SOFollowerRobotConfig(port=args.robot_port, id=args.robot_id)
        robot = robot_cls(robot_cfg)
        robot.connect()
        raw_obs = robot.get_observation()
        robot.disconnect()

    print(f"  Rohe Encoder-Werte vom Roboter:")
    for k, v in raw_obs.items():
        print(f"    {k:25s} = {v:.4f}")

    arm_pos_deg = np.array([float(raw_obs[f"{m}.pos"]) for m in ARM_JOINT_NAMES], dtype=np.float64)
    gripper_pct = float(raw_obs["gripper.pos"])
    gripper_rad = gripper_pct_to_rad(gripper_pct)
    gripper_deg = float(np.rad2deg(gripper_rad))
    joint_pos_deg = np.append(arm_pos_deg, gripper_deg)
    joint_pos_rad = np.deg2rad(joint_pos_deg)

    print(f"\n  Gelenkpositionen (Grad) nach Umrechnung:")
    for i, name in enumerate(ALL_JOINT_NAMES):
        print(f"    [{i}] {name:20s}: {joint_pos_deg[i]:8.3f}°  ({joint_pos_rad[i]:.4f} rad)")

    print(f"\n  Default-Positionen (Rad):")
    for i, name in enumerate(ALL_JOINT_NAMES):
        print(f"    [{i}] {name:20s}: {np.rad2deg(DEFAULT_JOINT_POS_RAD[i]):8.3f}°  ({DEFAULT_JOINT_POS_RAD[i]:.4f} rad)")

    joint_pos_rel = joint_pos_rad - DEFAULT_JOINT_POS_RAD
    print(f"\n  joint_pos_rel (= aktuell - default) in Rad:")
    for i, name in enumerate(ALL_JOINT_NAMES):
        print(f"    [{i}] {name:20s}: {joint_pos_rel[i]:+.4f} rad")

    # -- 3. FK: EE-Position --------------------------------------------------
    print(f"\n{SEP}")
    print("3. EE-POSITION VIA FK (bei aktuellem Roboter-State)")
    print(SEP)
    q_abs = {name: float(joint_pos_rad[i]) for i, name in enumerate(ALL_JOINT_NAMES)}
    ee_pos = get_ee_pos(q_abs, pin_model, pin_data, j_idx, gripper_frame_id)
    print(f"  EE-Position im Robot-Base-Frame:")
    print(f"    x = {ee_pos[0]:.4f} m")
    print(f"    y = {ee_pos[1]:.4f} m")
    print(f"    z = {ee_pos[2]:.4f} m")

    # EE auch bei Default-Position berechnen
    q_default = {name: float(DEFAULT_JOINT_POS_RAD[i]) for i, name in enumerate(ALL_JOINT_NAMES)}
    ee_pos_default = get_ee_pos(q_default, pin_model, pin_data, j_idx, gripper_frame_id)
    print(f"\n  EE-Position bei Default-Pose (Plausibilitaetscheck):")
    print(f"    x = {ee_pos_default[0]:.4f} m")
    print(f"    y = {ee_pos_default[1]:.4f} m")
    print(f"    z = {ee_pos_default[2]:.4f} m")
    print(f"  -> Erwarteter Bereich: x~0.15-0.25m, z~0.05-0.20m (grobe Schaetzung)")

    # -- 4. Observation-Vektor -----------------------------------------------
    print(f"\n{SEP}")
    print("4. OBSERVATION-VEKTOR (27D)")
    print(SEP)

    joint_vel_rel = np.zeros(6, dtype=np.float32)  # keine Vel verfuegbar

    bowl_with_offset = bowl_pos.copy()
    bowl_with_offset[2] += BOWL_HOVER_HEIGHT

    last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)

    obs_np = np.concatenate([
        joint_pos_rel.astype(np.float32),
        joint_vel_rel,
        ee_pos,
        object_pos,
        bowl_with_offset,
        last_action,
    ])

    labels = (
        [f"joint_pos_rel[{i}] ({ALL_JOINT_NAMES[i]})" for i in range(6)] +
        [f"joint_vel_rel[{i}] ({ALL_JOINT_NAMES[i]})" for i in range(6)] +
        ["ee_pos[x]", "ee_pos[y]", "ee_pos[z]"] +
        ["init_obj_pos[x]", "init_obj_pos[y]", "init_obj_pos[z]"] +
        ["bowl_pos[x]", "bowl_pos[y]", "bowl_pos[z]+0.12"] +
        [f"last_action[{i}]" for i in range(6)]
    )

    print(f"  {'Idx':>3}  {'Label':<35} {'Value':>10}")
    print(f"  {'---':>3}  {'-----':<35} {'-----':>10}")
    for i, (label, val) in enumerate(zip(labels, obs_np)):
        marker = ""
        if abs(val) > 5.0:
            marker = "  <- SUSPEKT (zu gross?)"
        elif i in range(12, 15) and abs(val) < 0.001:
            marker = "  <- SUSPEKT (EE nahe 0?)"
        print(f"  {i:>3}  {label:<35} {val:>10.4f}{marker}")

    print(f"\n  Uebergebene Positionen:")
    print(f"    --object_pos (initial_object_pos): {object_pos}")
    print(f"    --bowl_pos:                        {bowl_pos}")
    print(f"    bowl_pos mit z-Offset (+0.12m):    {bowl_with_offset}")

    # -- 5. Policy-Inferenz --------------------------------------------------
    print(f"\n{SEP}")
    print("5. POLICY-INFERENZ (10 Steps mit konstantem State)")
    print(SEP)

    policy = torch.jit.load(args.checkpoint, map_location="cpu")
    policy.eval()
    print(f"  Checkpoint geladen: {args.checkpoint}")

    obs_t = torch.from_numpy(obs_np).unsqueeze(0)  # (1, 27)
    print(f"  Obs-Tensor shape: {obs_t.shape}, dtype: {obs_t.dtype}")

    print(f"\n  Raw Policy-Outputs ueber 10 Steps (zeigt ob Policy konsistent ist):")
    print(f"  {'Step':>4}  {'pan':>7} {'lift':>7} {'elbow':>7} {'wrist_f':>7} {'wrist_r':>7} {'gripper':>7}")
    with torch.no_grad():
        for step in range(10):
            raw = policy(obs_t).squeeze(0).numpy()
            print(f"  {step:>4}  " + "  ".join(f"{v:>7.3f}" for v in raw))
            # last_action updaten (simuliert echten Loop)
            obs_np[21:27] = raw.astype(np.float32)
            obs_t = torch.from_numpy(obs_np).unsqueeze(0)

    # -- 6. Action-Interpretation (erster Step) ------------------------------
    print(f"\n{SEP}")
    print("6. ACTION-INTERPRETATION (Step 0)")
    print(SEP)

    # Obs neu aufbauen (ohne last_action-Update)
    obs_np[21:27] = 0.0
    obs_t = torch.from_numpy(obs_np).unsqueeze(0)
    with torch.no_grad():
        raw_action = policy(obs_t).squeeze(0).numpy()

    print(f"  Raw action: {raw_action.round(4)}")
    print(f"\n  Arm-Targets (default + 0.5 * raw_action):")
    for i, name in enumerate(ARM_JOINT_NAMES):
        target_rad = DEFAULT_JOINT_POS_RAD[i] + ARM_ACTION_SCALE * raw_action[i]
        target_deg = np.rad2deg(target_rad)
        current_deg = joint_pos_deg[i]
        delta = target_deg - current_deg
        print(f"    {name:20s}: default={np.rad2deg(DEFAULT_JOINT_POS_RAD[i]):7.2f}°  "
              f"raw={raw_action[i]:+.3f}  target={target_deg:7.2f}°  "
              f"current={current_deg:7.2f}°  delta={delta:+7.2f}°")

    gripper_target_rad = DEFAULT_JOINT_POS_RAD[5] + GRIPPER_ACTION_SCALE * raw_action[5]
    gripper_target_rad = float(np.clip(gripper_target_rad, GRIPPER_RAD_MIN, GRIPPER_RAD_MAX))
    print(f"  Gripper: default={np.rad2deg(DEFAULT_JOINT_POS_RAD[5]):.2f}°  "
          f"raw={raw_action[5]:+.3f}  target={np.rad2deg(gripper_target_rad):.2f}° "
          f"({gripper_rad_to_pct(gripper_target_rad):.1f}%)")

    # -- 7. Sanity-Checks ----------------------------------------------------
    print(f"\n{SEP}")
    print("7. SANITY-CHECKS")
    print(SEP)

    issues = []

    if np.all(np.abs(joint_pos_rel[:5]) < 0.05):
        print(f"  [OK] Roboter ist nahe der Default-Position (joint_pos_rel ~ 0)")
    else:
        print(f"  [??] Roboter ist NICHT in Default-Position – Policy erwartet Start nahe Default!")
        issues.append("Roboter nicht in Default-Pose beim Start")

    if abs(ee_pos_default[0]) < 0.05 and abs(ee_pos_default[1]) < 0.05 and abs(ee_pos_default[2]) < 0.05:
        print(f"  [!!] EE-Position bei Default ist nahe 0 – URDF/FK moeglicherweise falsch!")
        issues.append("EE-Position bei Default ~ 0 (FK-Problem)")
    else:
        print(f"  [OK] EE-Position bei Default sieht plausibel aus: {ee_pos_default.round(3)}")

    if np.any(np.abs(obs_np[:6]) > 3.14):
        print(f"  [!!] joint_pos_rel > pi – Encoder-Konversion moeglicherweise falsch!")
        issues.append("joint_pos_rel > pi")
    else:
        print(f"  [OK] joint_pos_rel im sinnvollen Bereich")

    obj_dist = np.linalg.norm(object_pos[:2] - ee_pos[:2])
    print(f"  [i] EE->Wuerfel XY-Distanz in Default-Pose: {obj_dist:.3f} m")
    if obj_dist > 0.5:
        print(f"    [??] Sehr gross – stimmt die --object_pos Messung?")
        issues.append(f"EE->Wuerfel Distanz sehr gross ({obj_dist:.2f}m)")

    if np.all(np.abs(raw_action) < 0.05):
        print(f"  [!!] Policy gibt sehr kleine Actions aus – Policy koennte falsch geladen sein!")
        issues.append("Policy-Output ~ 0")
    else:
        print(f"  [OK] Policy gibt nicht-triviale Actions aus")

    print(f"\n{'=' * 60}")
    if issues:
        print(f"GEFUNDENE PROBLEME:")
        for issue in issues:
            print(f"  [!!] {issue}")
    else:
        print(f"Keine offensichtlichen Probleme gefunden.")
        print(f"Wenn der Roboter trotzdem nicht funktioniert, pruefe:")
        print(f"  1. Stimmt --object_pos mit der echten Wuerfelposition ueberein?")
        print(f"  2. Ist das Koordinatensystem des URDF == Koordinatensystem der Messung?")
        print(f"  3. Entspricht die URDF der echten Kalibrierung des Roboters?")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()