import gym
import numpy as np

# Define settings
env = gym.make('MountainCar-v0')
NUMBER_OF_ITERATIONS = 1000
DISPLAY_EACH = 200
GRANULARITY = 20
LEARNING_RATE = 0.1
DISCOUNT = 1
explore_rate = 1
DECAY_EXPLORE_MIN_ITERATION = 1
DECAY_EXPLORE_MAX_ITERATION = (NUMBER_OF_ITERATIONS // 2)
DECAY_VALUE = explore_rate / (DECAY_EXPLORE_MAX_ITERATION - DECAY_EXPLORE_MIN_ITERATION)

# init q_values table
number_of_possible_actions = env.action_space.n
number_of_params_in_state = env.observation_space.shape[0]
table_size = [GRANULARITY for _ in range(number_of_params_in_state)]
table_size.append(number_of_possible_actions)
q_values = np.zeros(table_size)


def make_discrete_state(state):
    discrete_state = np.copy(state)
    for index, value in enumerate(state):
        max = env.observation_space.high[index]
        min = env.observation_space.low[index]
        discrete_state[index] = np.rint(np.interp(value, [min, max], [0, GRANULARITY - 1]))
    return tuple(discrete_state.astype(np.int))


def update_q_value(current_state, new_state, action, reward):
    best_future_state_q_value = np.min(q_values[new_state])

    current_q = q_values[current_state + (action,)]

    new_q_value = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * best_future_state_q_value)

    q_values[current_state + (action,)] = new_q_value


max_reward = 0
last_100_results = [False for _ in range(100)]
for iteration in range(NUMBER_OF_ITERATIONS):
    current_state = make_discrete_state(env.reset())
    done = False
    render = iteration % DISPLAY_EACH == 0 and iteration > 0

    if iteration % DISPLAY_EACH == 0:
        print(f'iteration: {iteration} explore_rate: {explore_rate} win rate over last 100 iterations: {sum(last_100_results)}%')

    while not done:

        noise = np.random.random()
        if noise <= explore_rate:
            action = np.random.randint(0, 3)

        else:

            min_value = np.amin(q_values[current_state])
            actions = np.argwhere(np.logical_and(q_values[current_state] >= min_value - 0.01,
                                                 q_values[current_state] <= min_value + 0.01))

            if actions.shape[0] != 1:
                action = np.random.choice(np.squeeze(actions), 1, replace=False)[0]
            else:
                action = actions[0][0]

        new_state, reward, done, _ = env.step(action)
        reward = abs(reward)
        max_reward = reward * 1.10 if reward > max_reward else max_reward

        if render:
            env.render()

        if not done:
            new_state = make_discrete_state(new_state)
            update_q_value(current_state, new_state, action, reward)
            current_state = new_state

        elif new_state[0] >= env.goal_position:
            q_values[current_state + (action,)] = -max_reward
            print(f'iteration win: {iteration}')
            last_100_results.pop(0)
            last_100_results.append(True)

        elif done:
            last_100_results.pop(0)
            last_100_results.append(False)

    if DECAY_EXPLORE_MAX_ITERATION >= iteration >= DECAY_EXPLORE_MIN_ITERATION:
        explore_rate -= DECAY_VALUE

env.close()
