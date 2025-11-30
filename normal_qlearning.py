import numpy as np
import matplotlib.pyplot as plt

n_actions = 4
gride_size = 5

n_states = gride_size * gride_size

Q_table = np.zeros((n_states, n_actions))

rewards = np.full(n_states, -1)
rewards[24] = 10
rewards[12] = -10

alpha = 0.1
gamma = 0.9
epsilon = 0.1

def epsilon_greedy_action(Q_table, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(0, n_actions)
    else:
        return np.argmax(Q_table[state])

for episode in range(1000):
    state = np.random.randint(0, n_states)
    done = False
    while not done:
        action = epsilon_greedy_action(Q_table, state, epsilon)
        next_state = np.random.randint(0, n_states)
        reward = rewards[state]
        Q_table[state, action] += alpha * (
            reward + gamma *np.max(Q_table[next_state]) - Q_table[state, action]
        )
        state = next_state
        if next_state in [12, 24]:
            done = True

culmelative_rewards = []
for episode in range(1000):
    state = np.random.randint(0, n_states)
    done = False
    total_rewards = 0
    while not done:
        action = epsilon_greedy_action(Q_table, state, epsilon)
        next_state = np.random.randint(0, n_states)
        reward = rewards[state]
        state = next_state
        total_rewards += reward
        if next_state in [12, 24]:
            done = True
        
    culmelative_rewards.append(total_rewards)

plt.plot(culmelative_rewards)
plt.title("Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Cumulative Rewards")
plt.show()

episode_length = []
for episode in range(1000):
    state = np.random.randint(0, n_states)
    done = False
    step = 0
    while not done:
        action = epsilon_greedy_action(Q_table, state, epsilon)
        next_state = np.random.randint(0, n_states)
        step += 1
        state = next_state
        if next_state in [12, 24]:
            done = True

    episode_length.append(step)

plt.hist(episode_length, bins=20)
plt.xlabel("Episode Length (Steps)")
plt.ylabel("Frequency")
plt.title("Distribution of Episode Length")
plt.show()

actions = []
def epsilon_greedy_action(Q_table, epsilon, state):
    if np.random.rand() < epsilon:
        actions.append('explore')
        return np.random.randint(0, Q_table.shape[1])
    else:
        actions.append('exploit')
        return  np.argmax(Q_table[state])

culmelative_rewards = []
for episode in range(1000):
    state = np.random.randint(0, n_states)
    done = False
    total_rewards = 0
    while not done:
        action = epsilon_greedy_action(Q_table, epsilon, state)
        next_state = np.random.randint(0, n_states)
        total_rewards += 1
        state = next_state
        if next_state in [12, 24]:
            done = True
    
    culmelative_rewards.append(total_rewards)

success_count = sum(1 for reward in culmelative_rewards if reward >= 10)
success_rate = success_count / len(culmelative_rewards)

exploration_count = sum(1 for action in actions if action == 'explore')
exploitation_count = sum(1 for action in actions if action == 'exploit')
exploration_exploitation_ratio = exploration_count / (exploration_count + exploitation_count)

print(f"Success Rate: {success_rate*100}%")
print(f"Exploration vs. Exploitation Ratio: {exploration_exploitation_ratio}")