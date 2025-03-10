import numpy as np
from abc import ABC, abstractmethod


class Environment(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        raise NotImplementedError


class AssetEnv(Environment):
    def __init__(self, initial_wealth, T, aversion_rate=0.01, riskless_return=0.05, risky_return={0.4: 0.5, 0.6: -0.3},
                 action_space=[0.2, 0.8]):
        self.aversion_rate = aversion_rate
        self.riskless_return = riskless_return
        self.risky_return = risky_return
        self.action_space = action_space
        self.initial_wealth = initial_wealth
        self.T = T

        self.state_space = {t: set() for t in range(T + 1)}
        self.transition_prob = {}
        self._generate_states_and_transitions()

    def _generate_states_and_transitions(self):
        self.transition_prob = {}
        self.state_space = {0: [self.initial_wealth]}

        for t in range(self.T):
            self.transition_prob[t] = {}
            self.state_space[t + 1] = []

            for wealth in self.state_space[t]:
                self.transition_prob[t][wealth] = {}

                for action in self.action_space:
                    self.transition_prob[t][wealth] = {action: [] for action in self.action_space}
                    for prob, risky_r in self.risky_return.items():
                        next_wealth = np.round(
                            wealth * (action * (1 + risky_r) + (1 - action) * (1 + self.riskless_return)), 3)

                        self.state_space.setdefault(t + 1, []).append(next_wealth)

                        done = (t == self.T - 1)
                        reward = 0 if not done else self._compute_reward(next_wealth)

                        self.transition_prob[t][wealth][action].append((prob, next_wealth, reward, done))

    def reset(self):
        self.t = 0
        self.s = self.initial_wealth
        return self.s

    def step(self, action):
        transitions = self.transition_prob[self.t][self.s][action]
        probs = [trans[0] for trans in transitions]
        chosen = np.random.choice(len(transitions), p=transitions)
        prob, next_state, reward, done = transitions[chosen]

        self.s, self.t = next_state, self.t + 1
        return next_state, reward, done, {'prob': prob}

    def cara_reward(self, wealth):
        return (-np.exp(-self.aversion_rate * wealth)) / self.aversion_rate


class SARSAAgent:
    def __init__(self, env, discount_factor=0.7, epsilon=0.05, alpha=0.1):
        self.env = env
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.alpha = alpha

        self.Q = {(t, s): {a: 0 for a in env.action_space} for t, states in env.state_space.items() for s in states}
        self.policy = {t: {s: self._uniform_policy(env.action_space) for s in states} for t, states in
                       env.state_space.items()}

    def _uniform_policy(self, actions):
        prob = 1 / len(actions)
        return [(prob, action) for action in actions]

    def select_action(self, t, s):
        actions, probs = zip(*self.policy[t][s])
        return np.random.choice(actions, p=probs)

    def improve_policy(self, t, s):
        best_action = max(self.Q[(t, s)], key=self.Q[(t, s)].get)
        policy_prob = []
        for action in self.Q[(t, s)]:
            if action == best_action:
                policy_prob.append((1 - self.env.action_dim * self.epsilon / len(self.Q[(t, s)]), action))
            else:
                policy_prob.append((self.epsilon / len(self.Q[(t, s)]), action))
        self.policy[t][s] = policy_prob

    def train(self, episodes=10000, verbose=True):
        for episode in range(episodes):
            s = self.env.reset()
            t = 0
            action = self.select_action(t, s)

            while True:
                next_s, reward, done, _ = self.env.step(action)

                if done:
                    break

                next_action = self.select_action(t + 1, next_s)

                td_target = reward + 0.7 * self.Q[(t + 1, next_s)][next_action]
                self.Q[(t, s)][action] += 0.1 * (td_target - self.Q[(t, s)][action])

                self.improve_policy(t, s)
                s, action, t = next_s, next_action, t + 1

            if verbose and (episode + 1) % 1000 == 0:
                print(f'Episode: {episode + 1}/{episodes}')

        return self.policy


if __name__ == '__main__':
    env = AssetEnv()
    sarsa_agent = SARSAAgent(env)
    policy = sarsa_agent.train()
    print(policy)
    print("111")