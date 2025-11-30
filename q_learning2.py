import numpy as np
import matplotlib.pyplot as plt

grid_size = 5
n_states = grid_size * grid_size

rewards = np.full(n_states, -1)
rewards[24] = 10
rewards[12] = -10

n_actions = 4

def epsilon_greedy_action(Q_table, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(0, n_actions)
    else:
        return np.argmax(Q_table[state])

alpha = 0.1
gamma = 0.9
epsilon = 0.1

Q_table = np.zeros((n_states, n_actions))

for episode in range(1000):
    state = np.random.randint(0, n_states)
    done = False

    while not done:
        action = epsilon_greedy_action(Q_table, state, epsilon)

        next_state = np.random.randint(0, n_states)

        reward = rewards[state]

        Q_table[state, action] += alpha * (
            reward + gamma * np.max(Q_table[next_state]) - Q_table[state, action]
        )

        state = next_state

        if next_state in [24, 12]:
            done = True

cumulative_rewards = []
for episode in range(1000):
    state = np.random.randint(0, n_states)
    total_reward = 0
    done = False
    while not done:
        action = epsilon_greedy_action(Q_table, state, epsilon)
        next_state = np.random.randint(0, n_states)
        reward = rewards[state]
        total_reward += reward
        state = next_state
        if next_state in [24, 12]:
            done = True

    cumulative_rewards.append(total_reward)

plt.plot(cumulative_rewards)
plt.title("Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.show()

episode_length = []
for episode in range(1000):
    step = 0
    state = np.random.randint(0, n_states)
    done = False
    while not done:
        action = epsilon_greedy_action(Q_table, state, epsilon)
        next_state = np.random.randint(0, n_states)
        step += 1
        state = next_state
        if next_state in [24, 12]:
            done = True

    episode_length.append(step)

plt.hist(episode_length, bins=20)
plt.xlabel("Episode Length (Steps)")
plt.ylabel("Frequency")
plt.title("Distribution of Episode Length")
plt.show()


# Redefine epsilon_greedy_action to log explorations & exploitations
actions = []
def epsilon_greedy_action(Q_table, state, epsilon):
    # Epsilon-greedy strategy: with probability epsilon, take a random action (exploration)
    # otherwise take the action with the highest Q-value for the given state (exploitation)
    if np.random.rand() < epsilon:  # Exploration
        actions.append('explore')
        return np.random.randint(0, Q_table.shape[1])  # Random action
    else:  # Exploitation
        actions.append('exploit')
        return np.argmax(Q_table[state])  # Action with the highest Q-value

# Calculate and store cumulative rewards and actions
cumulative_rewards = []
for episode in range(1000):
    total_reward = 0
    state = np.random.randint(0, n_states)
    done = False
    while not done:
        action = epsilon_greedy_action(Q_table, state, epsilon)
        next_state = np.random.randint(0, n_states)
        reward = rewards[next_state]
        total_reward += reward
        state = next_state
        if next_state == 24 or next_state == 12:
            done = True
    cumulative_rewards.append(total_reward)

success_count = sum(1 for reward in cumulative_rewards if reward >= 10)
success_rate = success_count / len(cumulative_rewards)


exploration_count = sum(1 for action in actions if action == 'explore')
exploitation_count = sum(1 for action in actions if action == 'exploit')
exploration_exploitation_ratio = exploration_count / (exploration_count + exploitation_count)

print(f"Success Rate: {success_rate * 100}%")
print(f"Exploration vs. Exploitation Ratio: {exploration_exploitation_ratio}")