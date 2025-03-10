import numpy as np
from abc import ABC, abstractmethod


class Environment(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass


class AssetEnv(Environment):
    def __init__(self, initial_wealth, T, aversion_rate=0.01, riskless_return=0.05,
                 risky_return={0.4: 0.5, 0.6: -0.3}, action_space=[0.2, 0.8]):
        self.aversion_rate = aversion_rate
        self.riskless_return = 0.05
        self.risky_return = risky_return
        self.action_space = action_space
        self.action_dim = len(action_space)
        self.initial_wealth = initial_wealth
        self.T = T

        self.state_space = {0: [initial_wealth]}
        self.transition_prob = {}
        self._generate_states_and_transitions()

    def _generate_states_and_transitions(self):
        for t in range(self.T):
            self.transition_prob[t] = {}
            self.state_space[t + 1] = []

            for wealth in self.state_space[t]:
                self.transition_prob[t][wealth] = {}
                for action in self.action_space:
                    self.transition_prob[t][wealth][action] = []

                    for prob, risky_r in self.risky_return.items():
                        next_wealth = np.round(
                            wealth * (action * (1 + risky_r) + (1 - action) * (1 + self.riskless_return)), 3)

                        if next_wealth not in self.state_space[t + 1]:
                            self.state_space[t + 1].append(next_wealth)

                        done = t == (self.T - 1)
                        reward = 0 if not done else self.cara_reward(next_wealth)
                        self.transition_prob[t][wealth][action].append((prob, next_wealth, reward, done))

    def reset(self):
        self.t, self.s = 0, self.initial_wealth
        return self.s

    def step(self, action):
        transitions = self.transition_prob[self.t][self.s][action]
        probs = [trans[0] for trans in transitions]
        chosen = np.random.choice(len(transitions), p=probs)
        prob, next_s, reward, done = transitions[chosen]

        self.s, self.t = next_s, self.t + 1
        return next_s, reward, done, {'prob': prob}

    def cara_reward(self, wealth):
        return -np.exp(-self.aversion_rate * wealth) / self.aversion_rate


class SARSAAgent:
    def __init__(self, env, discount_factor=0.7, epsilon=0.05):
        self.env = env
        self.epsilon = epsilon
        self.gamma = discount_factor
        self.alpha = 0.1
        self.Q = {(t, s): {a: 0 for a in env.action_space} for t in range(env.T) for s in env.state_space[t]}
        self.policy = {(t, s): np.random.choice(env.action_space) for t, s in self.Q}

    def sarsa_iteration(self, episodes=10000, verbose=True):
        for episode in range(episodes):
            s, t = self.env.reset(), 0
            action = self.policy[(t, s)]
            done = False

            while not done:
                next_s, reward, done, _ = self.env.step(action=self.policy[(t, s)])
                if done:
                    break
                next_action = self.policy[(t + 1, next_s)]
                td_target = reward + 0.7 * self.Q[(t + 1, next_s)][next_action]
                self.Q[(t, s)][action] += self.alpha * (td_target - self.Q[(t, s)][action])

                best_next_action = max(self.Q[(t, s)], key=self.Q[(t, s)].get)
                self.policy[(t, s)] = best_next_action
                s, action, t = next_s, next_action, t + 1

            if verbose and (episode + 1) % 1000 == 0:
                print(f'Episode: {episode + 1}/{episodes}')
        return self.policy


if __name__ == '__main__':
    env = AssetEnv(initial_wealth=10, T=10)
    agent = SARSAAgent(env)
    final_policy = agent.sarsa_iteration()
    print(final_policy)
    print("sss")