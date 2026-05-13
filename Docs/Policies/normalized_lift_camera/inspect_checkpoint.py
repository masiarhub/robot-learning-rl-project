import torch, sys

path = sys.argv[1] if len(sys.argv) > 1 else "model_2999.pt"
ckpt = torch.load(path, map_location="cpu", weights_only=False)

print(f"=== Checkpoint: {path} ===")
print(f"Top-level keys: {list(ckpt.keys())}")
print(f"Iteration: {ckpt.get('iter', '?')}")

sd = ckpt.get("model_state_dict", ckpt)
print(f"\n=== model_state_dict ({len(sd)} keys) ===")
for k, v in sd.items():
    print(f"  {k:60s}  {str(tuple(v.shape)):20s}  {v.dtype}")
