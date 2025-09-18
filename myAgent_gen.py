from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.agents.agent import Agent
from policy_gen import Policy, ValueFunctionApproximator
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
import numpy as np

class myAgent(Agent):
    def __init__(self, actor: Policy, critic: ValueFunctionApproximator, old_policy: Policy, idx: int, base_env: OvercookedEnv ):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.old_policy = old_policy
        self.idx = idx
        self.base_env = base_env

    def action(self, obs, layout_one_hot: np.ndarray) -> tuple[Action, dict]:
        if isinstance(obs, OvercookedState):
            feat_state = self.base_env.featurize_state_mdp(obs)
            obs = (feat_state[0], feat_state[1])

        flat_obs = np.concatenate([obs[0].flatten(), obs[1].flatten(), layout_one_hot])
        batch_obs = np.expand_dims(flat_obs, axis=0)

        action_prob = self.actor.call(batch_obs)[self.idx].numpy()
        action = Action.sample(np.squeeze(action_prob))

        return action, {'action_prob': action_prob}

    def update_old_policy(self):
        if self.old_policy is not None:
            self.old_policy.set_weights(self.actor.get_weights())
        else:
            print("No old policy to update")