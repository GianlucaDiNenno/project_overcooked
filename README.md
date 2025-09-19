## INTRODUCTION

Overcooked-AI is a benchmark environment based on the video game Overcooked, where two agents must collaboratively prepare and deliver meals as fast as possible.

 This project is built upon the Open Source Overcooked-AI repository (https://github.com/HumanCompatibleAI/overcooked_ai) which provides the implementation of the environment, including the logic of the game, the 
 representation of the state, and the basic interactions of the agents, such as actions and rewards.

 # RELEVANT FILES
 Relevant files in the project:
 
 - `training_gpu` : gpu accelerated implementation to train agents on single layout cramped_room
 - `training_incr.py`: gpu accelerated implementation to train agents on multiple layout. Incr because it save weights with increasing difficulty capability of solving difficult layouts
 - `train_two_policies.py` : imlementation of 2 different policies one for each agent.
 - `report.pdf`: pdf containg the project report
 - `weights/`: folder containg the weights of the models with the corresponding experimant name
 
