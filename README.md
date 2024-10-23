# SARSA Learning Algorithm

## AIM
To develop a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.

### States

The environment has 7 states:
* Two Terminal States: **G**: The goal state & **H**: A hole state.
* Five Transition states / Non-terminal States including  **S**: The starting state.

### Actions

The agent can take two actions:

* R: Move right.
* L: Move left.

### Transition Probabilities

The transition probabilities for each action are as follows:

* **50%** chance that the agent moves in the intended direction.
* **33.33%** chance that the agent stays in its current state.
* **16.66%** chance that the agent moves in the opposite direction.

For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

### Rewards

The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.

## SARSA LEARNING ALGORITHM
1. Initialize the Q-values arbitrarily for all state-action pairs.
2. Repeat for each episode:
    1. Initialize the starting state.
    2. Repeat for each step of episode:
        1. Choose action from state using policy derived from Q (e.g., epsilon-greedy).
        2. Take action, observe reward and next state.
        3. Choose action from next state using policy derived from Q (e.g., epsilon-greedy).
        4. Update Q(s, a) := Q(s, a) + alpha * [R + gamma * Q(s', a') - Q(s, a)]
        5. Update the state and action.
    3. Until state is terminal.
3. Until performance converges.
## SARSA LEARNING FUNCTION
### Name: Nathin R
### Register Number:212222230090
```
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):

    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))

    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    for e in tqdm(range(n_episodes), leave=False):
        state, done = env.reset(), False
        action = select_action(state, Q, epsilons[e])

        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = select_action(next_state, Q, epsilons[e])

            td_target = reward + gamma * Q[next_state, np.argmax(Q[next_state])] * (1 - done)
            td_error = td_target - Q[state, action]
            Q[state, action] += alphas[e] * td_error

            state = next_state
            action = next_action

        Q_track[e] = Q.copy()
        pi_track.append(np.argmax(Q, axis=1))

    V = np.max(Q, axis=1)
    pi = lambda s: np.argmax(Q[s])

    return Q, V, pi, Q_track, pi_track
```
```
print('Name:  Nathin R             Register Number: 212222230090            ')
print_state_value_function(V_sarsa, P, n_cols=n_cols, 
                           prec=svf_prec, title='State-value function found by Sarsa:')
print_state_value_function(optimal_V, P, n_cols=n_cols, 
                           prec=svf_prec, title='Optimal state-value function:')
print_state_value_function(V_sarsa - optimal_V, P, n_cols=n_cols, 
                           prec=err_prec, title='State-value function errors:')
print('State-value function RMSE: {}'.format(rmse(V_sarsa, optimal_V)))
print()
print_action_value_function(Q_sarsa, 
                            optimal_Q, 
                            action_symbols=action_symbols, 
                            prec=avf_prec, 
                            title='Sarsa action-value function:')
print('Action-value function RMSE: {}'.format(rmse(Q_sarsa, optimal_Q)))
print()
print_policy(pi_sarsa, P, action_symbols=action_symbols, n_cols=n_cols)
success_rate_sarsa, mean_return_sarsa, mean_regret_sarsa = get_policy_metrics(
    env, gamma=gamma, pi=pi_sarsa, goal_state=goal_state, optimal_Q=optimal_Q)
print('Reaches goal {:.2f}%. Obtains an average return of {:.4f}. Regret of {:.4f}'.format(
    success_rate_sarsa, mean_return_sarsa, mean_regret_sarsa))
```
## OUTPUT:
```
Name:  Nathin R             Register Number: 212222230090            
State-value function found by Sarsa:
|           | 01 0.5389 | 02 0.7533 | 03 0.8452 | 04 0.8896 | 05 0.9223 | 06  0.953 | 07 0.9826 |           |
Optimal state-value function:
|           | 01 0.5637 | 02  0.763 | 03 0.8449 | 04 0.8892 | 05  0.922 | 06 0.9515 | 07 0.9806 |           |
State-value function errors:
|           | 01  -0.02 | 02  -0.01 | 03    0.0 | 04    0.0 | 05    0.0 | 06    0.0 | 07    0.0 |           |
State-value function RMSE: 0.0089

Sarsa action-value function:
╒═════╤═══════╤═══════╤═══════╤═══════╤═════════╤═════════╕
│   s │     < │     > │   * < │   * > │   err < │   err > │
╞═════╪═══════╪═══════╪═══════╪═══════╪═════════╪═════════╡
│   0 │ 0     │ 0     │ 0     │ 0     │   0     │   0     │
├─────┼───────┼───────┼───────┼───────┼─────────┼─────────┤
│   1 │ 0.28  │ 0.539 │ 0.312 │ 0.564 │   0.032 │   0.025 │
├─────┼───────┼───────┼───────┼───────┼─────────┼─────────┤
│   2 │ 0.666 │ 0.753 │ 0.67  │ 0.763 │   0.004 │   0.01  │
├─────┼───────┼───────┼───────┼───────┼─────────┼─────────┤
│   3 │ 0.801 │ 0.845 │ 0.803 │ 0.845 │   0.002 │  -0     │
├─────┼───────┼───────┼───────┼───────┼─────────┼─────────┤
│   4 │ 0.864 │ 0.89  │ 0.864 │ 0.889 │  -0.001 │  -0     │
├─────┼───────┼───────┼───────┼───────┼─────────┼─────────┤
│   5 │ 0.903 │ 0.922 │ 0.901 │ 0.922 │  -0.001 │  -0     │
├─────┼───────┼───────┼───────┼───────┼─────────┼─────────┤
│   6 │ 0.933 │ 0.953 │ 0.932 │ 0.952 │  -0.001 │  -0.001 │
├─────┼───────┼───────┼───────┼───────┼─────────┼─────────┤
│   7 │ 0.961 │ 0.983 │ 0.961 │ 0.981 │   0.001 │  -0.002 │
├─────┼───────┼───────┼───────┼───────┼─────────┼─────────┤
│   8 │ 0     │ 0     │ 0     │ 0     │   0     │   0     │
╘═════╧═══════╧═══════╧═══════╧═══════╧═════════╧═════════╛
Action-value function RMSE: 0.01

Policy:
|           | 01      > | 02      > | 03      > | 04      > | 05      > | 06      > | 07      > |           |
Reaches goal 96.00%. Obtains an average return of 0.8672. Regret of 0.0000
```
![image](https://github.com/user-attachments/assets/541bc2ad-ac03-4507-89d0-49d6fe050f57)

![image](https://github.com/user-attachments/assets/c486ec08-7117-4cad-bfe6-00d30395c356)

## RESULT:

Thus the optimal policy for the given RL environment is found using SARSA-Learning and the state values are compared with the Monte Carlo method.
