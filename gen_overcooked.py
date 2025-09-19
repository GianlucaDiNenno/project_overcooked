from src.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from src.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
import random

class GeneralizedOvercooked:
    """
    Class to handle multiple Overcooked environments with different layouts.
    """
    def __init__(self, layouts, info_level=0, horizon=400):
        self.envs = []
        self.num_layouts = len(layouts)
        for layout in layouts:
            base_mdp = OvercookedGridworld.from_layout_name(layout)
            base_env = OvercookedEnv.from_mdp(mdp=base_mdp, info_level=info_level, horizon=horizon)
            env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)
            self.envs.append(env)
        self.cur_idx = 0
        self.current_env = self.envs[self.cur_idx]
        self.observation_space = self.current_env.observation_space
        self.action_space = self.current_env.action_space

    def reset(self):
        self.cur_idx = (self.cur_idx + 1) % len(self.envs) # iterate through the environments
        #self.cur_idx = random.randint(0, len(self.envs)-1) # alternatively, randomly select an environment
        self.current_env = self.envs[self.cur_idx]
        return self.current_env.reset()
    
    def get_layout_one_hot(self):
        """
        Returns a one-hot encoded representation of the current layout."""
        one_hot = [0] * self.num_layouts
        one_hot[self.cur_idx] = 1
        return one_hot
    
    def step(self, *args):
        return self.current_env.step(*args)
    
    def render(self, *args):
        return self.current_env.render(*args)

        