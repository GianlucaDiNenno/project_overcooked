from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.agents.agent import RandomAgent
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from policy_gen import Policy, ValueFunctionApproximator
from myAgent_gen import myAgent
import tensorflow as tf
import imageio.v2 as imageio
import pygame
import numpy as np
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
    parser.add_argument("--exp-name", type=str, default="GEN_exp_increasing_difficulty_5",
                        help="the name of the experiment used for loading the weights")
    parser.add_argument("--layout_name", type=str, default="cramped_room",
                        help="the name of the layout on which to evaluate the agents")
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="number of episodes for which to compute the average reward")
    parser.add_argument("--two-actors", type=bool, default=False,
                        help="whether to use one policy for each agent")
    parser.add_argument("--episode-render", type=int, default=4, help="episode number to render")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="entropy coefficient for the entropy bonus in the loss function")
    parser.add_argument("--render", type=lambda x: (str(x).lower() == "true"), default=False,
                        help="whether to render or not the first episode")

    args = parser.parse_args()

    return args

def load_weights():
    if TWO_ACTORS:
        actor_1.load_weights(PATH_ACTOR_1)
        actor_2.load_weights(PATH_ACTOR_2)
    else:
        actor.load_weights(PATH_ACTOR)
    print("")
    print("Weights successfully loaded.")
    print("")


def episode_to_video(states, base_mdp, out_path, fps=6, overlay_text_fn=None):
    """
    Render an episode into a video file GIF.
    - states: list[OvercookedState]
    - base_mdp: OvercookedGridworld
    - out_path: path to output video file 
    - fps: frames per second
    - overlay_text_fn: str for drawing an overlay per frame
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Headless-friendly: no need to open a window
    if not pygame.get_init():
      pygame.init()
    if not pygame.font.get_init():
      pygame.font.init()
    font = pygame.font.SysFont("Arial", 16)

    vis = StateVisualizer()
    frames_np = []

    for t, state in enumerate(states):
        # Render a pygame.Surface
        surf = vis.render_state(state, grid=base_mdp.terrain_mtx)

        # Optional overlay (timestep, rewards, etc.)
        if overlay_text_fn is not None:
            text = overlay_text_fn(t)
            if text:
                label = font.render(text, True, (0, 0, 0))
                surf.blit(label, (8, 8))

        # Convert Surface -> (H, W, 3) numpy array
        arr = pygame.surfarray.array3d(surf)        # (W, H, 3)
        frame = np.transpose(arr, (1, 0, 2))        # -> (H, W, 3)
        frames_np.append(frame)

    
    imageio.mimsave(out_path, frames_np, fps=fps)  # GIF, no ffmpeg needed
    try:
        #gif_fallback = os.path.splitext(out_path)[0] + ".gif"
        imageio.mimsave(out_path, frames_np, fps=fps)
        print(f"Saved GIF insteadon: {out_path}")
        return out_path
    except Exception as e2:
        raise RuntimeError(f"Failed to export video/GIF: {e2}")

    return out_path

if __name__ == "__main__":
    args = parse_args()

    # algorithm specifications
    EXP_NAME = args.exp_name
    LAYOUT = args.layout_name
    NUM_EPISODES = args.num_episodes
    RENDER = args.render
    TWO_ACTORS = args.two_actors
    EPISODE_RENDER = args.episode_render
    ENTROPY = args.entropy_coef

    PATH_ACTOR = os.path.join("weights", "actor", "actor_" + EXP_NAME + ".weights.h5")
    PATH_ACTOR_1 = os.path.join("weights", "actor", "actor_1_" + EXP_NAME + ".weights.h5")
    PATH_ACTOR_2 = os.path.join("weights", "actor", "actor_2_" + EXP_NAME + ".weights.h5")



    
    print("TF version:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    print("")
    print("EXPERIMENT INFO.")
    print(f"Experiment Name: {EXP_NAME}")
    print(f"Number of episodes: {NUM_EPISODES}")
    print("")

    #initialization environment
    horizon = 400
    layout_name = LAYOUT
    base_mdp = OvercookedGridworld.from_layout_name(layout_name=layout_name)
    base_env = OvercookedEnv.from_mdp(base_mdp, info_level=0, horizon=horizon)
    env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)
    env.render_mode = 'rgb_array'

    inp_shape = env.observation_space.shape[0]
    new_inp_shape = (inp_shape*2 + 5,)
    print(" Input shape:", new_inp_shape)


    if TWO_ACTORS:
        actor_1 = Policy(inp_shape=new_inp_shape, num_actions=Action.NUM_ACTIONS, entropy_coef=ENTROPY, optimizer=None)
        actor_2 = Policy(inp_shape=new_inp_shape, num_actions=Action.NUM_ACTIONS, entropy_coef=ENTROPY, optimizer=None)
        load_weights()
        agent_1 = myAgent(actor=actor_1, critic=None, old_policy=None, idx=0, base_env=base_env)
        agent_2 = myAgent(actor=actor_2, critic=None, old_policy=None, idx=1, base_env=base_env)
    else:
        actor = Policy(inp_shape=new_inp_shape, num_actions=Action.NUM_ACTIONS, entropy_coef=ENTROPY, optimizer=None)
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
            action_1 = agent_1.action(obs['both_agent_obs'], [0,0,0,0,1])
            action_2 = agent_2.action(obs['both_agent_obs'], [0,0,0,0,1])

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

            # if RENDER and episode == 2:
            #     img = env.render()

            #     frame_file = os.path.join("renders", f"episode_{episode}_frame_{t}.png")
            #     Image.fromarray(img).save(frame_file)

            t += 1
        aligned_sparse_rewards = episode_sparse_rewards[1:]
        cum_sparse = np.cumsum(aligned_sparse_rewards).tolist()
        overlay = lambda t: f"Timestep: {t}\nCumulative Shaped Reward: {cum_sparse[t]:.1f}"

        if RENDER and episode == EPISODE_RENDER:
            video_dir = os.path.join("renders")
            video_path = os.path.join(video_dir, f"{EXP_NAME}_episode_{episode}.gif")
            out_file = episode_to_video(states, base_mdp, video_path, fps=6, overlay_text_fn=overlay)
            print(f"Saved video of episode {episode} to {out_file}")


        cumulative_shaped_rewards.append(sum(episode_shaped_rewards))
        cumulative_sparse_rewards.append(sum(episode_sparse_rewards))

        average_shaped_reward = round(sum(cumulative_shaped_rewards)/len(cumulative_shaped_rewards), 3)
        average_sparse_reward = round(sum(cumulative_sparse_rewards)/len(cumulative_shaped_rewards), 3)

        end_episode = time.time()

        print(f"Episode [{episode:>3d}] terminated at timestep {t}. "
              f"cumulative sparse reward: {sum(episode_sparse_rewards):>3d}."
              f"cumulative shaped reward: {sum(episode_shaped_rewards):>3d}.")

    time.sleep(1)

    print("")
    print(f"Average results in {NUM_EPISODES} episodes:")
    print(f"avg sparse reward: {average_sparse_reward}. ")
    print(f"max sparse reward: {max(cumulative_sparse_rewards)}.")
    print(f"avg shaped reward: {average_shaped_reward}. ")
    print("")
