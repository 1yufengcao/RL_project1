# **README: SARSA-based Investment Optimization in Asset Allocation Environment**  

## **Overview**
This project implements a **SARSA (State-Action-Reward-State-Action) reinforcement learning agent** to optimize investment decisions in a simulated asset allocation environment. The environment models the evolution of an investor’s wealth over a finite time horizon, considering both **risk-free and risky assets**.

The agent learns an optimal investment strategy using **SARSA**, an on-policy reinforcement learning algorithm that updates the Q-values while following the same policy used for action selection.

---

## **Project Structure**
- **`AssetEnv`**: Defines the asset allocation environment where an investor decides the proportion of wealth to allocate to risky vs. risk-free assets.
- **`SARSAAgent`**: Implements the SARSA algorithm to learn an optimal policy for asset allocation.
- **`main`**: Runs the training loop, extracts the learned policy, and visualizes the results.

---

## **Installation & Dependencies**
Ensure you have **Python 3.7+** installed along with the required libraries:

```bash
pip install numpy matplotlib
```

---

## **Code Explanation**

### **1. Environment (`AssetEnv`)**
The **Asset Environment** models an investor making sequential investment decisions over a fixed time horizon (`T`). The investor can allocate wealth between:
- **Risk-free asset** with a fixed return (`riskless_return`).
- **Risky asset**, which provides a probabilistic return (`risky_return`).

#### **Key Components**
- **State Space** (`state_space`): Represents possible wealth values at each time step.
- **Action Space** (`action_space`): Defines investment choices (e.g., 0% or 100% in risky assets).
- **Transition Model** (`_generate_states_and_transitions`): Computes possible wealth transitions for each action.
- **Reward Function** (`cara_reward`): Defines the terminal wealth as the reward.

#### **Methods**
- `reset()`: Resets the environment to the initial state.
- `step(action)`: Executes an action, updates wealth, and returns the next state, reward, and whether the episode is done.

---

### **2. SARSA Agent (`SARSAAgent`)**
The **SARSA algorithm** learns an optimal policy for investment decisions.

#### **Key Components**
- **Q-Table** (`Q[t][wealth][action]`): Stores action-value estimates for each state-action pair.
- **Policy** (`policy[t][wealth]`): Defines action probabilities using an **ε-greedy strategy**.

#### **Methods**
- `_init_policy()`: Initializes a uniform random policy.
- `_update_policy()`: Updates the policy using **ε-greedy selection**.
- `choose_action()`: Selects an action based on the current policy.
- `train(episodes)`: Runs SARSA learning over multiple episodes.

---

### **3. Training & Visualization**
The **training loop** runs SARSA over multiple episodes, updating the Q-table and policy. After training, the learned policy is analyzed and visualized.

#### **Visualization**
A **line plot** shows the evolution of the **average optimal action over time**, helping to understand how investment decisions change across different stages.

```python
plt.plot(range(env.T), avg_actions, marker='o', linestyle='-', color='b')
plt.xlabel('Time')
plt.ylabel('Average Action (Risky Asset Allocation)')
plt.title('Average Optimal Action over Time (SARSA Policy)')
plt.grid(True)
plt.show()
```

---

## **Usage**
### **1. Running the Code**
Run the script to train the SARSA agent and visualize the learned investment policy:

```bash
python script.py
```

### **2. Modifying Parameters**
You can modify environment parameters in `AssetEnv`:

```python
env = AssetEnv(initial_wealth=10, T=10, aversion_rate=0.01, riskless_return=0.0002)
```

Adjust SARSA parameters in `SARSAAgent`:

```python
agent = SARSAAgent(env, discount_factor=1, epsilon=0.8, alpha=0.7)
```

---

## **Expected Output**
1. Text-based policy output mapping wealth levels to investment actions.
2. A **matplotlib graph** showing the **average action (risk allocation) over time**.

---

## **Future Improvements**
- Extend to **continuous action spaces**.
- Implement **Deep Q-Learning (DQN)** for more complex environments.
- Allow for **more complex risk-return distributions**.

---

## **Author & License**
- **Author**: [Your Name]
- **License**: MIT License
