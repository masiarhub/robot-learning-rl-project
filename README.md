# robot-learning-rl-project
Group Project 3: Singulation - Reinforcement Learning


#### MANUEL ####
Task 1:
Hybrid BC + RL 

Phase 1 (BC):
    Teleoperation Daten sammeln
    Supervised Learning
Phase 2 (RL):
    SAC (Soft Actor-Critic) bessere sample efficiency

Input: RGB-Bild (Wrist Camera)
Output: Robot Action (Greifen + Platzieren)
Umgebung: randomisierte Positionen


RGB Image
   ↓
CNN Encoder (ResNet18)
   ↓
Feature Vector
   ↓
+ Goal Input (target position xyz)
   ↓
MLP Policy Head
   ↓
Action (Δx, Δy, Δz, gripper)

Das selbe netzwerk wird zuerst mit BC trainiert und dann mit RL feingetuned. 

HIL-SERL -> https://huggingface.co/docs/lerobot/en/hilserl
ACT -> https://huggingface.co/docs/lerobot/act
isaac_so_arm101
robot-control-stack

Task 2:

Ist theoretisch das selbe wie task 1. 
BC + RL verwenden
Selbes Model wie für task 1 verwenden.
Farbmodel trainier das für eine gewünschte Farbe auf den richtigen würfel zeigt.
Einfache version mit tresholding (einfach, anfällig auf belichtungs unterschiede)
ML model das mit synthetischen daten tainiert wird (daten mit isaac lab generieren)

Für alle tasks:
Domain Randomization!!! Verkleienrt sim to real cap erheblich


https://github.com/MINT-SJTU/Evo-RL

### Leon
#### Eval 1
1) [LeRobot - Official Repo](https://github.com/huggingface/lerobot)

✅ Teleoperation data collection

✅ Dataset pipeline

✅ ACT (Action Chunking with Transformers) policy training for Eval 1 

✅ Real-arm deployment inference

-> seems to be quite straight-forward, most parts for Eval 1 already implemented 

useful resources:

[LeRobot Tutorial: Imitation Learning on Real-World Robots](https://huggingface.co/docs/lerobot/main/en/il_robots)

[Blog Post: ACT on SO101, Journey, Gotchas, and Lessons](https://huggingface.co/blog/sherryxychen/train-act-on-so-101)

[lerobot/svla_so101_pickplace: Dataset, das wir für ein Proof of Concept (hoffentlich) brauchen können](https://huggingface.co/datasets/lerobot/svla_so101_pickplace)

2) some custom implementation
- probably doesn't make sense unless we realize there is a major bottleneck with 1)

#### Eval 2 & 3
1) [MuammerBay - isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101)

+ mentioned by TAs in the task description PDF

+ ready to use basic framework

+ uses RSL-RL natively (which we will most probably use, TAs said it would be a good option)


- sim to real transfer: "work in progress"

- some tutorials available (see https://wiki.seeedstudio.com/training_soarm101_policy_with_isaacLab/), however: seems deprecated (uses older isaacsim/lab versions than newest codebase)

2) [Nvidia - 
Sim-to-Real-SO-101-Workshop
](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop)

+ brand new

+ official Isaac Lab tutorial

- brand new (potential bugs)

- specifically for use with GROOT VLA

  


## Masiar

### General Project Pipeline

### Simulation 
- [isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101)
- [adding a camera to the wrist](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/camera.html)
- [Randomization of block colors and position on the table](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html)
- [Setting color of block](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/sensors/omni_sensors_docs/materials_extension/materials_extension.html#current-materials)
- [IsaacLab's vectorized environment](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.html): For example using multiple environements in parallel.
- Sim-to-real gap mitigations: Random texture on table, random lighting intensity, slight camera noise, color jitter on block materials during training.

### Eval 1
#### Approach
- Behavior Cloning using lots of good quality demos --> Learning from Homework 2: Do not record f.e. 50 demos at once! Add iteratively more (always good quality!) demos: 1) record 5-10 demos 2) train policy on converted data 3) test/evaluate performance --> restart with demos if performance is still shit. 
- Recommendation of Claude for finetuing: Using soft-actor critic (SAC) from demonstrations for finetuing ("closing the generalization gap from randomize block positions").
- 
#### Architecture
ResNet-18 visual encoder (pretrained, frozen initially) + MLP actor. Input: wrist RGB image (84x84) + gripper state + bowl (x,y,z coordinates --> w.r.t world frame?). Bowl coordinates = goal vector.

#### Resources
- Obvious resource: [HuggingFace LeRobot](https://github.com/huggingface/lerobot)
- For the optional implementation of SAC: [SERL = "Sample-Efficient Robot RL"](seed a replay buffer with a handful of demos, run SAC on real/sim hardware)
- [Robot Learning Homework 3](https://github.com/mees-robot-learning-course/ethz-course-2026/tree/main/hw3_imitation_learning) 

### Eval 2
#### Approach
Modular perception + goal-conditioned SAC

1) Modular perception: A color segmentation head on top of our ResNet encoder (instead of creating and training model that detects and separates color end-to-end). Input: Wrist RGB image. Output: Pixel mask or bounding box of target block.
2) Goal-conditioned SAC (policy): Goal input is `embedded target color, (x,y,z)-coordinates of bowl`. Color embedding can be a one-hot vector since the color set is known or three-dimensional vector with RGB-values.
3) Training: Eval 1 policy as a start point. From there finetuning with new goal input head. Seed replay buffer with demos of both colors.
#### Resources


### Eval 3
#### Approach
Clutter-Handler + Sequencer + reusable sub-policy

- Handling clutter (own learned or fixed policy): 
> while `target color` is not detecable by wrist camera or `target color`'s surrounding is not empty:
>     1) Localize clutter or cubes in `target color's surrounding` w.r.t world(?) frame
>     2) Move to clutter position
>     3) Push/Hit (gently) the clutter 
- One reusable sub-policy (f.e. from Eval 2): goal-conditioned pick-and-place for a specified `(color, (x,y,z)-bowl-coordinate)`. An execution of a sub-policy is one step.
- Sequencer: Just a state-machine (not learned) that iterates through the goal list and call the sub-policy for each and detects the completion of a step.
#### Resources

