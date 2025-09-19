from src.overcooked_ai_py.mdp.actions import Action
from src.overcooked_ai_py.agents.agent import Agent
from policy_gpu import Policy, ValueFunctionApproximator
from src.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from src.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
import numpy as np

class myAgent(Agent):
    def __init__(self, actor: Policy, old_policy: Policy, critic: ValueFunctionApproximator, idx: int, base_env: OvercookedEnv ):
        super().__init__()
        self.actor = actor
        self.old_policy = old_policy
        self.idx = idx
        self.critic = critic
        self.base_env = base_env

    def action(self, obs) -> tuple[Action, dict]:
        """
        The function takes thas input the current observation (obs) 

        It returns the chosen action and a dictionary with action probabilities.
        """

        if isinstance(obs, OvercookedState):
            feat_state = self.base_env.featurize_state_mdp(obs)
            obs = (feat_state[0], feat_state[1])

        #extracting the action probabilities for the specific agent
        action_prob = self.actor.call(obs)[self.idx].numpy()
        action = Action.sample(np.squeeze(action_prob)) #sampling an action based on the probabilities

        return action, {'action_prob': action_prob}

    def update_old_policy(self):
        if self.old_policy is not None:
            self.old_policy.set_weights(self.actor.get_weights())
        else:
            print("No old policy to update")