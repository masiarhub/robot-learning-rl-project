# robot-learning-rl-project
Group Project 3: Singulation - Reinforcement Learning

### Leon
#### Eval 1
1) https://github.com/huggingface/lerobot (base repo that we will use anyway)

✅ Teleoperation data collection
✅ Dataset pipeline
✅ ACT (Action Chunking with Transformers) policy training for Eval 1 
✅ Real-arm deployment inference

-> seems to be quite straight-forward, most parts for Eval 1 already implemented (reference: https://huggingface.co/blog/sherryxychen/train-act-on-so-101)

2) some custom implementation
- doesn't make sense unless we realize there is a major bottleneck with 1)

#### Eval 2 & 3
1) https://github.com/MuammerBay/isaac_so_arm101

+ mentioned by TAs in the task description PDF
+ ready to use basic framework
+ uses RSL-RL natively (which we will most probably use, TAs said it would be a good option)

- sim to real transfer: "work in progress"
- some tutorials available (see https://wiki.seeedstudio.com/training_soarm101_policy_with_isaacLab/), however: seems deprecated (uses older isaacsim/lab versions than newest codebase)

2) https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop

+ brand new
+ official Isaac Lab tutorial

- brand new (potential bugs)
- specifically for use with GROOT VLA
