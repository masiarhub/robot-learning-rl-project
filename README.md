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

https://huggingface.co/docs/lerobot/act
https://github.com/MINT-SJTU/Evo-RL



