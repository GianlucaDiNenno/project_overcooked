from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.agents.agent import RandomAgent
from policy_gpu import Policy, ValueFunctionApproximator
from myAgent import myAgent
import tensorflow as tf
import argparse
from PIL import Image
import time
import os


def parse_args():
    """
    Parse command line arguments for the experiment configuration.

    Returns:
        args (Namespace): Parsed command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="test_GPU_GAE" \
    "_exp", help="the name of the experiment used for loading the weights")
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="number of episodes for which to compute the average reward")
    parser.add_argument("--render", type=lambda x: (str(x).lower() == "true"), default=False,
                        help="whether to render or not the first episode")

    args = parser.parse_args()

    return args

def load_weights():
    actor.load_weights(PATH_ACTOR)
    print("")
    print("Weights successfully loaded.")
    print("")

if __name__ == "__main__":
    args = parse_args()

    # algorithm specifications
    EXP_NAME = args.exp_name
    NUM_EPISODES = args.num_episodes
    RENDER = args.render

    PATH_ACTOR = os.path.join("weights", "actor", "actor_" + EXP_NAME + ".weights.h5")
    
    print("TF version:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    print("")
    print("EXPERIMENT INFO.")
    print(f"Experiment Name: {EXP_NAME}")
    print(f"Number of episodes: {NUM_EPISODES}")
    print("")

    #initialization environment
    horizon = 400
    layout_name = "cramped_room"
    base_mdp = OvercookedGridworld.from_layout_name(layout_name=layout_name)
    base_env = OvercookedEnv.from_mdp(base_mdp, info_level=0, horizon=horizon)
    env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)
    env.render_mode = 'rgb_array'

    inp_shape = env.observation_space.shape

    if EXP_NAME == "random_agents":
        agent_1 = RandomAgent(all_actions=True)
        agent_2 = RandomAgent(all_actions=True)
    else:
        actor = Policy(inp_shape=inp_shape, num_actions=Action.NUM_ACTIONS, optimizer=None)
        load_weights()
        agent_1 = myAgent(actor=actor, critic=None, old_policy=None, idx=0, base_env=base_env)
        agent_2 = myAgent(actor=actor, critic=None, old_policy=None, idx=1, base_env=base_env)

    cumulative_sparse_rewards = []  # list of cumulative sparse rewards for each episode
    cumulative_shaped_rewards = []  # list of cumulative shaped rewards for each episode

    # running the episodes
    for episode in range(1, NUM_EPISODES + 1):
        states = []

        t = 0
        obs = env.reset()
        done = False

        episode_sparse_rewards = [0]
        episode_shaped_rewards = [0]

        start = time.time()

        while not done:
            action_1 = agent_1.action(obs['both_agent_obs'])
            action_2 = agent_2.action(obs['both_agent_obs'])

            agent_1_action_idx = Action.ACTION_TO_INDEX[action_1[0]]
            agent_2_action_idx = Action.ACTION_TO_INDEX[action_2[0]]

            action = (agent_1_action_idx, agent_2_action_idx)

            states.append(obs['overcooked_state'])

            new_obs, reward, done, env_info = env.step(action)

            shaped_rewad = sum(env_info['shaped_r_by_agent'])
            sparse_reward = reward
            total_reward= shaped_rewad + sparse_reward

            episode_sparse_rewards.append(sparse_reward)
            episode_shaped_rewards.append(total_reward)

            obs = new_obs

            # rendering the episode
            if RENDER and episode == 1:
                img = env.render()

                frame_file = os.path.join("renders", f"episode_{episode}_frame_{t}.png")
                Image.fromarray(img).save(frame_file)

            t += 1


        cumulative_shaped_rewards.append(sum(episode_shaped_rewards))
        cumulative_sparse_rewards.append(sum(episode_sparse_rewards))

        average_shaped_reward = round(sum(cumulative_shaped_rewards)/len(cumulative_shaped_rewards), 3)
        average_sparse_reward = round(sum(cumulative_sparse_rewards)/len(cumulative_shaped_rewards), 3)

        end_episode = time.time()

        print(f"Episode [{episode:>3d}] terminated at timestep {t}. "
              f"cumulative sparse reward: {sum(episode_sparse_rewards):>3d}."
              f"cumulative shaped reward: {sum(episode_shaped_rewards):>3d}.")


    print("")
    print(f"Average results in {NUM_EPISODES} episodes:")
    print(f"avg sparse reward: {average_sparse_reward}. ")
    print(f"max sparse reward: {max(cumulative_sparse_rewards)}.")
    print(f"avg shaped reward: {average_shaped_reward}. ")
    print("")
