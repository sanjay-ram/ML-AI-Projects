import tensorflow as tf
import numpy as np

grid_size = 5
# Define the policy network
n_states = grid_size * grid_size  # Number of states in the grid
n_actions = 4  # Up, down, left, right

model = tf.keras.Sequential([
    tf.keras.layers.Dense(24, activation='relu', input_shape=(n_states,)),
    tf.keras.layers.Dense(n_actions, activation='softmax')  # Output action probabilities
])

# Optimizer for policy network updates
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

def get_action(state):
    state_input = tf.one_hot(state, n_states)  # One-hot encoding for state
    action_probs = model(state_input[np.newaxis, :])
    return np.random.choice(n_actions, p=action_probs.numpy()[0])

rewards = np.full((grid_size * grid_size,), -1)  # -1 for every state
rewards[24] = 10  # Goal state
rewards[12] = -10  # Pitfall state


# Simulation loop
states = []
actions = []
episode_rewards = []  

for episode in range(1000):
    state = np.random.randint(0, n_states)  # Start in a random state
    done = False
    while not done:
        action = get_action(state)  # Use the provided function
        next_state = np.random.randint(0, n_states)  # Simulated next state
        reward = rewards[next_state]  

        # Store the state-action-reward trajectory
        states.append(state)
        actions.append(action)
        episode_rewards.append(reward)  

        state = next_state
        if next_state in {24, 12}:  
            done = True
    def compute_cumulative_rewards(rewards, gamma=0.99):
        cumulative_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * gamma + rewards[t]
            cumulative_rewards[t] = running_add
        return cumulative_rewards
    def update_policy(states, actions, rewards):
        cumulative_rewards = compute_cumulative_rewards(rewards)

        with tf.GradientTape() as tape:
            state_inputs = tf.one_hot(states, n_states)  # Convert states to one-hot encoding
            action_probs = model(state_inputs)
            action_masks = tf.one_hot(actions, n_actions)  # Mask for selected actions
            log_probs = tf.reduce_sum(action_masks * tf.math.log(action_probs), axis=1)

        # Policy loss is the negative log-probability of the action times the cumulative reward
            loss = -tf.reduce_mean(log_probs * cumulative_rewards)

        # Apply gradients to update the policy network
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))