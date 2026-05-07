"""
Per-eval deployment entrypoint for Project 3.

This wraps `deploy_sim.py` (the squint deploy script, copied unchanged so the
sim2real bridge through LeRobotRealAgent stays intact) with a single argument:
which eval to run. Each eval pins the env id and the goal-conditioning kwargs.

Usage examples
--------------
Eval 1 — single block, single bowl pinned at TA-supplied (x,y):
    python deploy_sim_eval.py --eval=1 --checkpoint=runs/eval1/ckpt.pt \
        --target_bowl_pos 0.30 0.05

Eval 2 — targeted pick-and-place, target color = red, bowl pinned:
    python deploy_sim_eval.py --eval=2 --checkpoint=runs/eval2_targeted/ckpt.pt \
        --target_color_idx=0 --target_bowl_pos 0.30 0.05

Eval 3 — sequential, reuses Eval 2 checkpoint via sequential_runner:
    python deploy_sim_eval.py --eval=3 --checkpoint=runs/eval2_targeted/ckpt.pt \
        --bowl_positions  0.32 -0.10  0.32 0.00  0.32 0.10 \
        --sequence 0 1 2
"""

from __future__ import annotations

import argparse
import sys
from typing import List


def _parse_pairs(values: List[str], pair_size: int = 2):
    """Parse a flat list like [x1, y1, x2, y2, ...] into a list of tuples."""
    if len(values) % pair_size != 0:
        raise ValueError(f"Expected multiples of {pair_size} values; got {len(values)}")
    return [tuple(float(v) for v in values[i : i + pair_size])
            for i in range(0, len(values), pair_size)]


def main():
    p = argparse.ArgumentParser(description="Project 3 per-eval deploy wrapper")
    p.add_argument("--eval", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to ckpt.pt from train_sim.py")
    # Eval 1 / Eval 2 args
    p.add_argument("--target_bowl_pos", nargs=2, type=float, default=None,
                   metavar=("X", "Y"))
    p.add_argument("--target_color_idx", type=int, default=None)
    # Eval 3 args
    p.add_argument("--bowl_positions", nargs="+", type=float, default=None,
                   help="Flat list of x1 y1 x2 y2 x3 y3 (3 bowls).")
    p.add_argument("--sequence", nargs=3, type=int, default=None,
                   help="Three color indices: e.g. 0 1 2 for red->blue->green")
    # Pass-through args to deploy_sim.py
    p.add_argument("--max_episode_steps", type=int, default=None)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--continuous_eval", dest="continuous_eval", action="store_true",
                   default=True)
    p.add_argument("--no_continuous_eval", dest="continuous_eval", action="store_false")
    args = p.parse_args()

    # Build the argv we forward to deploy_sim.main(). deploy_sim.py uses tyro,
    # so we just construct sys.argv[1:] and re-invoke its main().
    forward = ["--checkpoint", args.checkpoint]
    if args.debug:
        forward.append("--debug")
    if not args.continuous_eval:
        forward.append("--no-continuous_eval")

    if args.eval == 1:
        env_id = "SO101PlaceBowlCubeFixed-v1"
        forward += ["--env_id", env_id]
        if args.target_bowl_pos is not None:
            forward += ["--target_bowl_pos",
                        f"{args.target_bowl_pos[0]}", f"{args.target_bowl_pos[1]}"]
        if args.max_episode_steps is None:
            args.max_episode_steps = 120
        forward += ["--max_episode_steps", str(args.max_episode_steps)]

    elif args.eval == 2:
        env_id = "SO101TargetedPlaceFixed-v1"
        forward += ["--env_id", env_id]
        if args.target_color_idx is None:
            raise SystemExit("Eval 2 requires --target_color_idx")
        if args.target_bowl_pos is None:
            raise SystemExit("Eval 2 requires --target_bowl_pos X Y")
        forward += [
            "--target_color_idx", str(args.target_color_idx),
            "--target_bowl_pos", f"{args.target_bowl_pos[0]}", f"{args.target_bowl_pos[1]}",
        ]
        if args.max_episode_steps is None:
            args.max_episode_steps = 150
        forward += ["--max_episode_steps", str(args.max_episode_steps)]

    elif args.eval == 3:
        # Eval 3 doesn't go through deploy_sim.py; we drive it via the
        # sequential_runner so the same Eval-2 checkpoint is reused 3 times.
        return _run_eval3(args)

    # For Eval 1 and 2, hand off to deploy_sim.py
    print(f"[deploy_sim_eval] Running eval {args.eval}: env_id={env_id}")
    print(f"[deploy_sim_eval] forwarding: {' '.join(forward)}")
    sys.argv = ["deploy_sim.py"] + forward
    import deploy_sim  # noqa: E402  (relative to the sim/ dir)
    deploy_sim.main() if hasattr(deploy_sim, "main") else None


def _run_eval3(args):
    """Reuse the Eval 2 checkpoint as a per-step sub-policy."""
    if args.bowl_positions is None or args.sequence is None:
        raise SystemExit("Eval 3 requires --bowl_positions x1 y1 x2 y2 x3 y3 and --sequence c0 c1 c2")

    bowl_positions = _parse_pairs(args.bowl_positions, pair_size=2)
    if len(bowl_positions) != 3:
        raise SystemExit("Eval 3 requires exactly 3 bowl positions")

    import gymnasium as gym
    import torch

    # Register envs
    import envs  # noqa: F401  (registers SO101* envs)

    # Build the multi-block sim env (use sim for evaluation by default; for
    # real-robot you would similarly wrap it through Sim2RealEnv, but the
    # sequential subgoal advancement requires the *sim* env's evaluate() so
    # we run it here in pure sim.)
    env = gym.make(
        "SO101MultiBlockSeq-v1",
        num_envs=1,
        obs_mode="rgb+segmentation",
        sensor_configs=dict(width=128, height=128),
        bowl_positions=bowl_positions,
        sequence_color_idx=tuple(args.sequence),
        domain_randomization=False,
    )

    # Load Eval 2 checkpoint via the same DeployAgent as squint's deploy.py
    from train_sim import DeployAgent  # noqa: E402

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    agent = DeployAgent.from_state_dict(env, ckpt)  # type: ignore[attr-defined]
    agent_act = lambda obs: agent.get_eval_action(obs["rgb"], obs["state"])  # noqa: E731

    from policies.sequential_runner import run_sequential
    obs, infos, _ = run_sequential(env, agent_act, max_steps=args.max_episode_steps or 300)
    print(f"[eval3] step_done = {infos.get('step_done')}, success = {infos.get('success')}")


if __name__ == "__main__":
    main()
