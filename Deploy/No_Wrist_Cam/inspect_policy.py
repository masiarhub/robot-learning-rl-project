# -*- coding: utf-8 -*-
import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
args = parser.parse_args()

policy = torch.jit.load(args.checkpoint, map_location="cpu")
policy.eval()

mean = None
var  = None
std  = None
for name, buf in policy.named_buffers():
    if "_mean" in name:
        mean = buf.squeeze().numpy()
    elif "_var" in name:
        var  = buf.squeeze().numpy()
    elif "_std" in name:
        std  = buf.squeeze().numpy()

labels = (
    [f"joint_pos_rel[{i}]" for i in range(6)] +
    [f"joint_vel_rel[{i}]" for i in range(6)] +
    ["ee_pos[x]", "ee_pos[y]", "ee_pos[z]"] +
    ["init_obj_pos[x]", "init_obj_pos[y]", "init_obj_pos[z]"] +
    ["bowl_pos[x]",  "bowl_pos[y]",  "bowl_pos[z]+0.12"] +
    [f"last_action[{i}]" for i in range(6)]
)

print(f"\n{'Idx':>3}  {'Label':<22}  {'mean':>10}  {'var':>12}  {'std':>10}  {'note'}")
print("-" * 75)
for i, label in enumerate(labels):
    m = mean[i] if mean is not None else float("nan")
    v = var[i]  if var  is not None else float("nan")
    s = std[i]  if std  is not None else float("nan")
    note = ""
    if v < 1e-6:
        note = "<-- VAR~0: normalizer will divide by ~0 !"
    elif v > 1000:
        note = "<-- very high variance"
    print(f"{i:>3}  {label:<22}  {m:>10.4f}  {v:>12.4f}  {s:>10.4f}  {note}")

# Now test: manually apply normalizer to typical obs
DEFAULT_RAD = np.array([0.00, -0.40, -0.30, 1.57, -1.57, 0.20], dtype=np.float32)
ee_pos   = np.array([0.2481, -0.0102, 0.2063], dtype=np.float32)
obj_pos  = np.array([0.23, 0.09, 0.00], dtype=np.float32)
bowl_pos = np.array([0.43, 0.00, 0.12], dtype=np.float32)
obs = np.concatenate([
    np.zeros(6), np.zeros(6), ee_pos, obj_pos, bowl_pos, np.zeros(6)
]).astype(np.float32)

if mean is not None and std is not None:
    eps = 1e-8
    safe_std = np.where(std < eps, 1.0, std)  # replace zero-std with 1.0
    obs_normed = (obs - mean) / safe_std
    print(f"\nNormalized obs (zero-std dims clamped to 1.0):")
    for i, (label, v) in enumerate(zip(labels, obs_normed)):
        print(f"  [{i:>2}] {label:<22}: {v:>10.4f}")

    # Run policy with safe-normalized obs
    obs_t = torch.from_numpy(obs_normed).unsqueeze(0)
    with torch.no_grad():
        # bypass internal normalizer by passing already-normalized obs
        # instead call actor directly
        out = obs_t
        for name, module in policy.named_modules():
            if "actor" in name and hasattr(module, "forward") and name == "actor":
                out = module(obs_t)
                break
    print(f"\nActor output with manually safe-normalized obs:")
    try:
        print(f"  {out.squeeze(0).numpy().round(4)}")
    except:
        print("  (could not extract actor output directly)")

    print("\nConclusion:")
    zero_dims = [labels[i] for i in range(27) if (std is not None and std[i] < 1e-6)]
    print(f"  Zero-variance dims: {zero_dims}")
    print(f"  These dims cause inf/nan in normalization -> exploding policy output.")
    print(f"  Fix: pass already-normalized obs to policy (bypass internal normalizer),")
    print(f"       OR retrain with these dims handled correctly.")