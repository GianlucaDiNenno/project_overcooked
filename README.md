## INTRODUCTION

Overcooked-AI is a benchmark environment based on the video game Overcooked, where two agents must collaboratively prepare and deliver meals as fast as possible.

 This project is built upon the Open Source Overcooked-AI repository (https://github.com/HumanCompatibleAI/overcooked_ai) which provides the implementation of the environment, including the logic of the game, the 
 representation of the state, and the basic interactions of the agents, such as actions and rewards.

 # RELEVANT FILES
 Relevant files in the project:
 
 - `training_gpu` : gpu accelerated implementation to train agents on single layout cramped_room
 - `training_gen_incr.py`: gpu accelerated implementation to train agents on multiple layout. Incr because it save weights with increasing difficulty capability of solving difficult layouts
 - `train_gen_two_policies.py` : implementation of 2 different policies one for each agent.
 - `gen_overcooked.py`: file containing the class GeneralizedOvercooked
 - `myAgent_gen.py`: file containg myAgent class with action() function
 - `policy_gen.py`: file containing the policy and value function network implementation for a general layout.
 - `policy_gpu.py`: file containing the policy and value function network implementation for a single layout.
 - `report.pdf`: pdf containg the project report
 - `weights/`: folder containg the weights of the critic and actor with the corresponding experiment name
 
