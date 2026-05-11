# Visualize SO-101 ManiSkill3 simulation tasks.
#
# GPU mode (default):  python visualize_sim.py
# Headless/CPU mode:   python visualize_sim.py --headless

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import argparse
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging
logging.disable(level=logging.WARN)

import numpy as np
import gymnasium as gym

# Add tasks
import envs
import mani_skill.envs


# =============================================================================
# Configuration
# =============================================================================

GPU_CONFIG = {
    'tasks': [
        'SO101ReachCube-v1', 'SO101ReachCan-v1',
        'SO101LiftCube-v1', 'SO101LiftCan-v1',
        'SO101PlaceCube-v1', 'SO101PlaceCan-v1',
        'SO101StackCube-v1', 'SO101StackCan-v1',
    ],
    'num_envs': 16,
    'sim_backend': 'auto',
    'obs_mode': 'rgb+segmentation',
    'render_mode': 'rgb_array',
    'image_size': 128,
    'downsample_size': 128,
    'color_jitter': False,
    'control_mode': None,
    'domain_randomization': True,
    'shader_dir': 'default',
    'window_size': 512,
    'steps_per_task': 30,
    'reset_interval': 10,
    'seed': 1,
}

HEADLESS_CONFIG = {
    'tasks': [
        'SO101ReachCube-v1',
        'SO101LiftCube-v1',
        'SO101PlaceBowlCube-v1',
        'SO101TargetedPlace-v1',
    ],
    'num_envs': 1,
    'sim_backend': 'cpu',
    # obs_mode='state' and render_mode=None bypass all Vulkan camera calls.
    'obs_mode': 'state',
    'render_mode': None,
    'control_mode': None,
    'domain_randomization': False,
    'steps_per_task': 20,
    'seed': 1,
}


# =============================================================================
# Environment Factory
# =============================================================================

def make_env(task: str, config: dict):
    env_kwargs = dict(
        obs_mode=config['obs_mode'],
        render_mode=config['render_mode'],
        num_envs=config['num_envs'],
        sim_backend=config.get('sim_backend', 'auto'),
        domain_randomization=config['domain_randomization'],
        reconfiguration_freq=None,
    )

    if config.get('control_mode') is not None:
        env_kwargs['control_mode'] = config['control_mode']

    if config.get('shader_dir') is not None:
        env_kwargs['shader_dir'] = config['shader_dir']

    if config.get('image_size') is not None:
        sensor_size = {'width': config['image_size'], 'height': config['image_size']}
        env_kwargs['sensor_configs'] = sensor_size
        env_kwargs['human_render_camera_configs'] = sensor_size

    env = gym.make(task, **env_kwargs)

    if 'rgb' in config['obs_mode']:
        import utils
        from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
        env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
        if config.get('downsample_size') is not None:
            env = utils.DownsampleObsWrapper(env, target_size=config['downsample_size'])
        if config.get('color_jitter'):
            env = utils.ColorJitterWrapper(env)

    env.reset(seed=config['seed'])
    return env


# =============================================================================
# Runners
# =============================================================================

def run_visual(config: dict):
    import cv2
    import torch
    from mani_skill.utils.visualization.misc import tile_images

    for task in config['tasks']:
        print(f"Instantiating: {task}")
        env = make_env(task, config)
        obs, info = env.reset()
        action_shape = env.action_space.shape
        num_envs = config['num_envs']
        video_nrows = int(np.sqrt(num_envs))
        window_size = config['window_size']
        steps_per_task = config['steps_per_task']
        reset_interval = config['reset_interval']

        print(f"Running: {task}")
        for step in range(steps_per_task):
            action = np.zeros(action_shape)
            action[..., -1] = 1 if step < 20 else -1

            obs, reward, terminated, truncated, info = env.step(action)
            done = (terminated | truncated).any()
            render_rgb = env.render()

            if isinstance(obs, dict) and 'rgb' in obs:
                obs_rgb = obs['rgb']
                if obs_rgb.shape[-1] != 3 and obs_rgb.shape[-1] % 3 == 0:
                    obs_rgb = obs_rgb[..., :3]
                render_h, render_w = render_rgb.shape[1], render_rgb.shape[2]
                if obs_rgb.shape[1] != render_h or obs_rgb.shape[2] != render_w:
                    obs_rgb = torch.nn.functional.interpolate(
                        obs_rgb.permute(0, 3, 1, 2).float(),
                        size=(render_h, render_w),
                        mode='nearest',
                    ).permute(0, 2, 3, 1).to(torch.uint8)
                paired = torch.cat([obs_rgb, render_rgb], dim=2)
                rgb = tile_images(paired, nrows=video_nrows).cpu().numpy().astype(np.uint8)
                rgb = cv2.resize(rgb, dsize=(window_size * 2, window_size))
            else:
                rgb = tile_images(render_rgb, nrows=video_nrows).cpu().numpy().astype(np.uint8)
                rgb = cv2.resize(rgb, dsize=(window_size, window_size))

            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            print(f"Step: {step}/{steps_per_task}, done={done}", end="\r")
            cv2.imshow("Obs | Render", rgb)
            cv2.waitKey(30)

            if (step % reset_interval == 0) or done:
                env.reset()

        env.close()
        cv2.destroyAllWindows()
        print(f"Finished: {task}                    ")



def run_headless(config: dict):
    passed, failed = [], []

    for task in config['tasks']:
        print(f"\n[{task}] instantiating...", flush=True)
        try:
            env = make_env(task, config)
            obs, info = env.reset()
            action_shape = env.action_space.shape

            for step in range(config['steps_per_task']):
                action = np.zeros(action_shape)
                action[..., -1] = 1 if step < 10 else -1
                obs, reward, terminated, truncated, info = env.step(action)
                done = bool((terminated | truncated).any())
                print(f"  step {step+1:02d}/{config['steps_per_task']}  reward={float(reward.mean()):.4f}  done={done}", flush=True)
                if done:
                    env.reset()

            env.close()
            print(f"[{task}] PASSED", flush=True)
            passed.append(task)

        except Exception as e:
            print(f"[{task}] FAILED — {e}", flush=True)
            failed.append(task)

    print(f"\n{'='*50}")
    print(f"Results: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print("Failed tasks:")
        for t in failed:
            print(f"  - {t}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--headless', action='store_true',
        help='Run headless physics-only smoke test (no GPU/Vulkan required).'
    )
    args = parser.parse_args()

    if args.headless:
        run_headless(HEADLESS_CONFIG)
    else:
        run_visual(GPU_CONFIG)
