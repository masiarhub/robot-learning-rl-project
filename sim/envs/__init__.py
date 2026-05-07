# Custom robots
import envs.robot.so100  # noqa: F401
import envs.robot.so101  # noqa: F401

# Original squint envs (kept available for ablations)
import envs.reach   # noqa: F401
import envs.lift    # noqa: F401
import envs.stack   # noqa: F401

# Project 3 envs
import envs.place_bowl           # SO101PlaceBowlCube-v1, SO101PlaceBowlCubeFixed-v1
import envs.targeted_pick_place  # SO101TargetedPlace-v1, SO101TargetedPlaceFixed-v1
import envs.multi_block_eval     # SO101MultiBlockSeq-v1

# NOTE: The original squint Place env (SO101PlaceCube-v1, SO101PlaceCan-v1) is
# preserved as `_place_original.py` for reference but NOT auto-registered, to
# avoid name collisions and to keep the project's "place into bowl" semantics
# canonical. To enable it, add: `import envs._place_original`
