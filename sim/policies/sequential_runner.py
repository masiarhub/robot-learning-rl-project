"""
Policy-switching scheduler for Eval 3 (Option B).

Reuses an Eval-2 (`SO101TargetedPlace*`) checkpoint and applies it three times
in sequence to a `SO101MultiBlockSeq-v1` env. The scheduler:

  1. Reads the env's `step_idx` to know which subgoal is active.
  2. Builds a goal-conditioning vector (target_color one-hot + target_bowl_pos)
     from the active subgoal.
  3. Patches the obs's `target_color_one_hot` and `target_bowl_pos` keys so the
     trained Eval 2 policy receives the right goal.
  4. Calls the policy. The env's evaluate() advances `step_idx` whenever a
     subgoal completes, so the next env step sees the next goal.

This file does NOT load the policy itself — it expects an `act(obs) -> action`
callable, so you can use the SQuint DeployAgent class straight from
`sim.train_sim.DeployAgent`, or any policy that exposes `.get_eval_action(rgb, state)`.

Typical usage:

    from sim.policies.sequential_runner import run_sequential
    obs, infos = run_sequential(env, policy_act_fn, max_steps=300)

The env's `info["step_done"]` is what determines per-step credit.
"""

from __future__ import annotations
from typing import Callable, Dict, Any

import torch


def patch_obs_for_subgoal(obs: Dict[str, Any], env) -> Dict[str, Any]:
    """In-place patch of an obs dict so that the goal-conditioning matches
    the env's *current* subgoal. This lets a policy trained for a single
    (color, bowl) goal be reused across the 3-step sequence."""
    # The multi-block env already places the current subgoal's color one-hot
    # and target bowl XY into the observation -- this function is a no-op for
    # `SO101MultiBlockSeq-v1`, but kept here so callers can drop in custom
    # envs that need explicit patching.
    return obs


def run_sequential(
    env,
    policy_act_fn: Callable[[Dict[str, Any]], torch.Tensor],
    *,
    max_steps: int = 300,
    return_video: bool = False,
):
    """Roll out the policy on a multi-block-sequential env.

    Args:
        env: a gym env produced by gym.make("SO101MultiBlockSeq-v1", ...). Must
            be vectorized (num_envs >= 1).
        policy_act_fn: callable mapping a single-step observation dict to a
            batched action tensor of shape (num_envs, action_dim). Typically
            something like:
                fn = lambda o: agent.get_eval_action(o["rgb"], o["state"])
        max_steps: hard cap on total env steps across the whole sequence.

    Returns:
        Last observation, last info dict (which includes per-step success).
    """
    obs, _ = env.reset()
    infos: Dict[str, Any] = {}
    frames = []
    for _ in range(max_steps):
        obs = patch_obs_for_subgoal(obs, env)
        with torch.no_grad():
            action = policy_act_fn(obs)
        obs, _, terminated, truncated, infos = env.step(action)
        if return_video:
            try:
                frames.append(env.render())
            except Exception:
                pass
        if torch.all(terminated | truncated):
            break
    return obs, infos, (frames if return_video else None)
