import tensorflow as tf
from keras import layers
import keras.optimizers as optimizers
from keras.layers import Input
from keras.models import Model

class Policy(Model):
    def __init__(self, inp_shape, num_actions, optimizer, entropy_coef:float, epsilon: float = 0.05):
        super().__init__()
        self.inp_shape = inp_shape
        self.num_actions = num_actions
        self.optimizer = optimizer
        self.entropy_coef = entropy_coef   
        self.epsilon = epsilon
        self.dense_1 = layers.Dense(64, activation='tanh')
        self.dense_2 = layers.Dense(128, activation='tanh')
        self.dense_3 = layers.Dense(64, activation='tanh')
        self.policy_a = layers.Dense(self.num_actions, activation='softmax', name = 'policy_a')
        self.policy_b = layers.Dense(self.num_actions, activation='softmax', name = 'policy_b')
        self.build_model()


    def call(self, obs, training: bool = False) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Defines the forward pass of the policy function network (the actor).

        Args:
            obs (tf.Tensor): The batch of observations (state features).
            training (bool): A flag indicating if the model is in training mode .

        Returns:
            tuple[tf.Tensor, tf.Tensor]: The predicted action probabilities of each agent.        
        """

        x = self.dense_1(obs)
        x = self.dense_2(x)
        x = self.dense_3(x) 
        policy_a = self.policy_a(x)
        policy_b = self.policy_b(x)

        return (policy_a, policy_b)

    def build_model(self):
        """
            Function to initialize the model by calling it with a dummy input
            This is necessary to create the model's weights before training or inference.
        """
        dummy_input = tf.zeros((1,) + self.inp_shape)
        
        _ = self(dummy_input)

    @tf.function
    def batch_train_PPO(self, delta_batch: tf.Tensor, obs_batch: tf.Tensor, actions_batch: tf.Tensor, old_policy: "Policy"):
        """
        Performs a single training step for the actor network using the PPO clipped surrogate objective with an entropy bonus.
        
        Args:
            delta_batch (tf.Tensor): The advantages calculated for each timestep (A_t).
            obs_batch (tf.Tensor): The observations for the batch (s_t).
            actions_batch (tf.Tensor): The actions taken in the batch (a_t).
            old_policy (Policy): The policy network before the updates, used to calculate the probability ratio.
        """

        if tf.rank(delta_batch) == 1:
            delta_batch = tf.stack([delta_batch, delta_batch], axis=1)

        with tf.GradientTape() as tape:
            # Get action probabilities for the batch from both the current policy and the old policy.            
            pi_a, pi_b = self.call(obs_batch, training=True)
            old_pi_a, old_pi_b = old_policy.call(obs_batch)

            # Get the probabilities of the actions that were actually taken by agent A.
            p_a = tf.gather(pi_a, actions_batch[:, 0], axis=1, batch_dims=1) 
            op_a = tf.gather(old_pi_a, actions_batch[:, 0], axis=1, batch_dims=1) 
            r_a = p_a / (op_a + 1e-8) # Calculate the probability ratio
            adv_a = tf.squeeze(delta_batch[:, :1], axis=1) # Advantage for agent A

            obj_a = r_a * adv_a # Calculate the standard, unclipped PPO objective.
            obj_a_clip = tf.clip_by_value(r_a, 1 - self.epsilon, 1 + self.epsilon) * adv_a # Calculate the clipped PPO objective.
            term_a = tf.minimum(obj_a, obj_a_clip)

            p_b = tf.gather(pi_b, actions_batch[:, 1], axis=1, batch_dims=1)  
            op_b = tf.gather(old_pi_b, actions_batch[:, 1], axis=1, batch_dims=1) 
            r_b = p_b / (op_b + 1e-8)  
            adv_b = tf.squeeze(delta_batch[:, 1:], axis=1)  

            obj_b = r_b * adv_b
            obj_b_clip = tf.clip_by_value(r_b, 1 - self.epsilon, 1 + self.epsilon) * adv_b
            term_b = tf.minimum(obj_b, obj_b_clip)

            #ENTROPY LOSS
            # Calculates the entropy of the probability distributions for both action spaces.
            # Entropy loss encourages exploration by penalizing deterministic policies.
            entropy_a = -tf.reduce_sum(pi_a * tf.math.log(pi_a + 1e-8), axis=1)
            entropy_b = -tf.reduce_sum(pi_b * tf.math.log(pi_b + 1e-8), axis=1)
            tot_entropy = tf.reduce_mean(entropy_a + entropy_b)
        
            ppo_loss = -tf.reduce_mean(term_a + term_b) 
            loss = ppo_loss - self.entropy_coef * tot_entropy # Combine PPO loss with entropy bonus

        grad_loss = tape.gradient(loss, self.trainable_weights) # Calculate the gradients of the loss with respect to the model's trainable weights.
        self.optimizer.apply_gradients(zip(grad_loss, self.trainable_weights)) # Apply the gradients to update the weights of the network.



class ValueFunctionApproximator(Model):
    def __init__(self, inp_shape, optimizer):
        super().__init__()
        self.inp_shape = inp_shape
        self.optimizer = optimizer
        self.dense_1 = layers.Dense(64, activation='tanh')
        self.dense_2 = layers.Dense(128,activation='tanh')
        self.dense_3 = layers.Dense(64, activation='tanh')
        self.value_function = layers.Dense(1, name='value_function')
        self.build_model()


    def call(self, obs, training: bool = False) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Defines the forward pass of the value function network (the critic).
        
        Args:
            obs (tf.Tensor): The batch of observations (state features).
            training (bool): A flag indicating if the model is in training mode .

        Returns:
            tf.Tensor: The predicted state-value for each observation in the batch.
        """

        x = self.dense_1(obs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        value = self.value_function(x)

        return value

    def build_model(self):
        """
            Function to initialize the model by calling it with a dummy input
            This is necessary to create the model's weights before training or inference.
        """
        dummy_input = tf.zeros((1,) + self.inp_shape)     
        _ = self(dummy_input)


    def batch_train_PPO(self, values_targets_batch, obs_batch):
        """
        Performs a single training step for the critic network.
        The goal is to minimize the Mean Squared Error between the predicted values and the target values.

        Args:
            values_targets_batch (tf.Tensor): The target values (GAE + V(s)) the critic should learn to predict.
            obs_batch (tf.Tensor): The batch of observations corresponding to the targets.
        """
        with tf.GradientTape() as tape:
            # The value targets are treated as the "ground truth". We use tf.stop_gradient
            # to ensure that no gradients are computed through them
            target = tf.stop_gradient(values_targets_batch)
            value = self.call(obs_batch) # Predicted values from the critic

            # Squeeze the output to ensure its shape (batch_size,) matches the target's shape.
            if len(value.shape) == 2:
                value = tf.squeeze(value, axis=1)

            error = target - value # Calculate the prediction error
            loss = tf.reduce_mean(tf.square(error)) # Mean Squared Error loss

        grad_loss = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grad_loss, self.trainable_weights))
