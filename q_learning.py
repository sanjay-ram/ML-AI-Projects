import numpy as np

grid_size = 5
n_actions = 4 

Q_table = np.zeros((grid_size, grid_size, n_actions))

n_states = grid_size * grid_size
rewards = np.full(n_states, -1)
rewards[24] = 10
rewards[12] = -10
def epsilon_greedy_action(Q_table, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(0, n_actions)
    else:
        x, y = state // grid_size, state % grid_size
        return np.argmax(Q_table[x, y])

alpha = 0.1
gamma = 0.9
epsilon = 0.1

for episode in range(1000):
    state = np.random.randint(0, n_states)
    done = False
    while not done:
        action = epsilon_greedy_action(Q_table, state, epsilon)

        next_state = np.random.randint(0, n_states)

        reward = rewards[state]

        x, y = state // grid_size, state % grid_size
        next_x, next_y = next_state // grid_size, next_state % grid_size

        Q_table[x, y, action] += alpha * (
            reward + gamma * np.max(Q_table[next_x, next_y]) - Q_table[x, y, action]
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

episode_length = []
for episode in range(1000):
    state = np.random.randint(0, n_states)
    step = 0
    done = False
    while not done:
        action = epsilon_greedy_action(Q_table, state, epsilon)
        next_state = np.random.randint(0, n_states)
        state = next_state
        step += 1
        if next_state in [24, 12]:
            done = True

    episode_length.append(step)

import matplotlib.pyplot as plt
plt.plot(cumulative_rewards)
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Learning Curve')
plt.show()
plt.plot(episode_length)
plt.xlabel('Episode')
plt.ylabel('Episode Length')
plt.title('Episode Length over Time')
plt.show()