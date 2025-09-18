from policy_gen import Policy, ValueFunctionApproximator
from myAgent_gen import myAgent
from gen_overcooked import GeneralizedOvercooked
from overcooked_ai_py.mdp.actions import Action
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import Adam
from typing import Tuple, List, Dict
import sys
import argparse
import json
import time
import os


def parse_args():
    """
    Command line arguments for the experiment configuration.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="GEN_at_once_not_random", help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=42, help="set the seed for reproducibility of the experiment")
    parser.add_argument("--num-episodes", type=int, default=5000, help="number of episodes to train the agent on")
    parser.add_argument("--num-epochs", type=int, default=4, help="number of epochs to train the agent on with SGD")
    parser.add_argument("--batch-size", type=int, default=256, help="batch size of the training with SGD")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="entropy coefficient for the entropy bonus in the loss function")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor for rewards and future state value estimations")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="lambda parameter for GAE")
    parser.add_argument("--lr-c", type=float, default=3e-4, help="learning rate for the critic")
    parser.add_argument("--lr-a", type=float, default=3e-4, help="learning rate for the actor")
    parser.add_argument("--load-weights", type=lambda x: (str(x).lower() == "true"), default=True, help="whether to load the weights of a previous experiment")
    parser.add_argument("--loading_file", type=str, default="", help="name of the file from which to load the weights")
    parser.add_argument("--saving_file", type=str, default="GEN_at_once_not_random", help="name of the file in which to save the weights")
    parser.add_argument("--ppo-epsilon", type=float, default=0.2, help="epsilon for clipping in PPO.")

    args = parser.parse_args()

    return args

def compute_advantages_gae(rewards: List[float], values: tf.Tensor, next_values: tf.Tensor, dones: List[bool], gamma: float, gae_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute advantages and value targets using GAE
    """

    advantages = np.zeros_like(rewards, dtype=np.float32) # Initialize advantages array
    last_adv = 0 # Variable to store the last computed advantage for GAE calculation

    values = tf.squeeze(values).numpy() 
    next_values = tf.squeeze(next_values).numpy()

    # Iterate over the rewards in reverse order to compute advantages using GAE
    for t in reversed(range(len(rewards))):
        if dones[t]: 
            #if state is terminal, next state value is 0
            next_val = 0
        else:
            next_val = next_values[t] # Value of the next state

        delta = rewards[t] + gamma*next_val - values[t] # TD error
        last_adv = delta + gamma*gae_lambda*last_adv # GAE formula
        advantages[t] = last_adv # Store the computed advantage

    value_targets = advantages + values 

    return advantages, value_targets

def load_weights():
    """
    Load the weights of the networks from a file.
    """
    if LOAD_WEIGHTS:
        actor.load_weights(PATH_ACTOR_LOAD)
        critic.load_weights(PATH_CRITIC_LOAD)
        print("")
        print("Weights successfully loaded.")
        print("")
    

def save_weights():
    """
    Save the weights of the networks to a file.
    If the file already exists, it will be overwritten.
    """
    try:
        critic.save_weights(PATH_CRITIC_SAVE)
        actor.save_weights(PATH_ACTOR_SAVE)
        print("Weights successfully saved.")
    except:
        print("Error occurred saving weights.")



def set_seed(SEED:int):
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    args = parse_args()

    # algorithm specifications
    EXP_NAME = args.exp_name
    SEED = args.seed
    LOAD_WEIGHTS = args.load_weights
    LOAD_FILE = args.loading_file
    SAVE_FILE = args.saving_file

    # hyperparameters
    LR_CRITIC = args.lr_c
    LR_ACTOR = args.lr_a
    NUM_EPISODES = args.num_episodes
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    GAMMA = args.gamma
    GAE_LAMBDA = args.gae_lambda
    PPO_EPSILON = args.ppo_epsilon
    ENTROPY = args.entropy_coef

    # penalties and rewards
    USELESS_DROP_PENALTY = -0.1
    USEFUL_PICKUP_REWARD = 0.1
    USEFUL_DROP_REWARD = 0.1
    OPTIMAL_POTTING_REWARD = 0.5
    CATASTROPHIC_POTTING_PENALTY = -0.5
    SOUP_PICKUP_REWARD = 0.5
    SOUP_DELIVERY = 1.0

    # paths for saving and loading weights and experiment info
    PATH_ACTOR_LOAD = os.path.join("weights", "actor", "actor_" + LOAD_FILE + ".weights.h5")
    PATH_ACTOR_SAVE = os.path.join("weights", "actor", "actor_" + SAVE_FILE + ".weights.h5")
    PATH_CRITIC_LOAD = os.path.join("weights", "critic", "critic_" + LOAD_FILE + ".weights.h5")
    PATH_CRITIC_SAVE = os.path.join("weights", "critic", "critic_" + SAVE_FILE + ".weights.h5")

    set_seed(SEED)

    # initializing the generalized environment
    horizon = 400
    layouts_name = ["cramped_room", "asymmetric_advantages", "coordination_ring", "forced_coordination", "counter_circuit"]
    gen_env = GeneralizedOvercooked(layouts=layouts_name, horizon=horizon)

    # initializing shape of the input for the networks
    agent_obs_shape = gen_env.observation_space.shape[0]
    new_inp_shape = (agent_obs_shape*2 + gen_env.num_layouts,)

    # defining the actor and critic
    actor = Policy(
        inp_shape=new_inp_shape,
        num_actions=Action.NUM_ACTIONS,
        optimizer=Adam(learning_rate=LR_ACTOR),
        entropy_coef=ENTROPY,
        epsilon=PPO_EPSILON
    )

    critic = ValueFunctionApproximator(
        inp_shape=new_inp_shape,
        optimizer=Adam(learning_rate=LR_CRITIC)
    )

    # old policy for PPO
    old_policy = Policy(inp_shape=new_inp_shape, 
                        num_actions=Action.NUM_ACTIONS, 
                        optimizer=None, entropy_coef=ENTROPY)
    second_critic = critic

    load_weights()


    #rollout phase
    for episode in range(1,NUM_EPISODES+1):
        start = time.time()

        #initializing lists to store episode information
        episode_rewards = []
        episode_observations = []
        episode_new_observations = []
        episode_actions = []
        episode_dones = []

        t = 0
        # resetting the environment
        obs = gen_env.reset()
        obs = obs['both_agent_obs']

        # getting the one-hot encoding of the layout
        layout_one_hot = gen_env.get_layout_one_hot()

        agent_1 = myAgent(
        actor=actor,
        old_policy=old_policy,
        critic=critic,
        idx=0,
        base_env=gen_env.current_env.base_env
        )

        agent_2 = myAgent(
            actor=actor,
            old_policy=old_policy,
            critic=second_critic,
            idx=1,
            base_env=gen_env.current_env.base_env
        )

        done = False
        ep_reward = 0

        # running the episode
        while not done:
            flat_obs = np.concatenate([obs[0].flatten(), obs[1].flatten(), layout_one_hot])

            action_1 = agent_1.action(obs, layout_one_hot)
            action_2 = agent_2.action(obs, layout_one_hot)

            agent_1_action_idx = Action.ACTION_TO_INDEX[action_1[0]]
            agent_2_action_idx = Action.ACTION_TO_INDEX[action_2[0]]

            action = (agent_1_action_idx, agent_2_action_idx)

            episode_actions.append(action)
            episode_observations.append(flat_obs)

            new_obs, reward, done, env_info = gen_env.step(action)
            new_obs = new_obs['both_agent_obs']

            flat_new_obs = np.concatenate([new_obs[0].flatten(), new_obs[1].flatten(), layout_one_hot])

            shaped_reward = env_info['shaped_r_by_agent'][0] + env_info['shaped_r_by_agent'][1]
            total_reward = reward + shaped_reward
            ep_reward += total_reward

            episode_rewards.append(total_reward)
            episode_new_observations.append(flat_new_obs)
            episode_dones.append(done)

            obs = new_obs

            t += 1
        
        #collecting information about the episode                
        final_stats = env_info.get('episode',{}).get('ep_game_stats',{})
    
        events_to_reward = {
            'useful_onion_pickup': USEFUL_PICKUP_REWARD,
            'useful_dish_pickup': USEFUL_PICKUP_REWARD,
            'useful_onion_drop': USEFUL_DROP_REWARD,
            'useful_dish_drop': USEFUL_DROP_REWARD,
            'optimal_onion_potting': OPTIMAL_POTTING_REWARD,
            'useless_onion_potting': USELESS_DROP_PENALTY,
            'catastrophic_onion_potting': CATASTROPHIC_POTTING_PENALTY,
            'soup_pickup': SOUP_PICKUP_REWARD,
            'soup_delivery': SOUP_DELIVERY,
        }

        # adding additional rewards and penalties based on events
        for event, reward in events_to_reward.items():
            timesteps_event = final_stats.get(event,[[],[]])
            for agent in range(len(timesteps_event)):
                for timestep in timesteps_event[agent]:
                    # adding reward to the specific timestep
                    if timestep <= horizon: #making sure the timestep is within the horizon
                        episode_rewards[timestep-1] += reward

        end_episode = time.time()

        #computing advantages and value targets
        state_values = critic.call(np.array(episode_observations))
        next_state_values = critic.call(np.array(episode_new_observations))

        advantages, value_targets = compute_advantages_gae(episode_rewards, state_values, next_state_values, episode_dones, GAMMA, GAE_LAMBDA)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        #preparing the batched dataset
        dataset = tf.data.Dataset.from_tensor_slices((episode_observations, episode_actions, advantages, value_targets))
        dataset = dataset.shuffle(buffer_size=len(episode_observations))
        dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        #training
        start_training = time.time()
        for epoch in range(1, NUM_EPOCHS+1):
           
            for obs_batch, actions_batch, advantages_batch, value_targets_batch in dataset:
                actor.batch_train_PPO(advantages_batch, obs_batch, actions_batch, old_policy)
                critic.batch_train_PPO(value_targets_batch, obs_batch)

        end_training = time.time()
        print(f"Episode {episode} cumulative reward: {ep_reward:.2f}, Training ended in {round(end_training - start_training, 2)} seconds")

        agent_1.update_old_policy()
        
        if episode % 50 == 0:
            save_weights()