import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

# 该代码实现了一个资产投资环境 (AssetEnv) 和一个基于 SARSA 算法的智能体 (SARSAAgent)。
# AssetEnv 模拟投资者在有风险和无风险资产之间分配财富的过程，定义了投资的状态转移和奖励计算。
# SARSAAgent 则在该环境中通过试错与环境交互进行学习，不断更新其对各状态下不同动作的价值估计 (Q值) 和策略 (policy)，
# 目标是学习一套投资策略，使得最终财富的效用期望最大化 (即在风险厌恶的前提下，实现财富增长的最佳策略)。
# 整个问题可以表示为一个有限时长的马尔可夫决策过程 (MDP)，状态包括时间和财富，动作为投资比例，随机转移由风险资产收益决定。

# 定义一个抽象类 Environment，表示通用的环境接口
# 该类规定了环境必须实现的两个基本方法: reset() 和 step()
class Environment(ABC):
    @abstractmethod
    def reset(self):
        # abstractmethod 装饰器表示这是一个抽象方法，必须在子类中实现
        # 重置环境到初始状态，并返回该初始状态
        pass

    @abstractmethod
    def step(self, action):
        # abstractmethod: 子类必须实现此方法来定义在执行动作后的环境变化
        # 此方法应返回 (新的状态, 奖励, 是否结束, 其他信息)
        pass

# 定义具体的资产投资环境类 AssetEnv，继承自 Environment 抽象基类
# 该环境模拟一个有无风险资产选择的投资过程:
#   - 投资者每一期可以选择将财富按一定比例投入有风险资产 (其余投入无风险资产)
#   - 无风险资产每期有固定收益率 riskless_return
#   - 有风险资产的收益率是不确定的，用 risky_return 字典给出若干可能结果及其概率
#   - 投资过程进行 T 期后结束，最终获得的财富根据风险厌恶型效用函数 (CARA) 计算得到奖励
class AssetEnv(Environment):
    def __init__(self, initial_wealth=10, T=10, aversion_rate=0.01, riskless_return=0.3,
                 risky_return=None, action_space=None):
        # initial_wealth: 初始财富
        # T: 总时间步数 (投资期数)
        # aversion_rate: 风险厌恶系数 (用于效用函数计算奖励)
        # riskless_return: 无风险资产每期固定收益率 (如 0.3 表示每期30%的收益率)
        # risky_return: 有风险资产的收益率分布 (字典: 概率 -> 收益率)，若为 None 则使用默认值
        # action_space: 动作空间 (可供选择的有风险资产投资比例列表)，若为 None 则默认为 [0, 1]
        # 以上参数确定了环境的初始条件和特性。
        # AssetEnv 将利用这些参数构建状态空间，并模拟财富随时间的演化过程。
        if risky_return is None:
            risky_return = {0.4: -1.5, 0.6: 1.5}  # 默认的概率到收益率的映射
            # 这里默认假设:
            #  - 40%的概率下，有风险资产收益率为 -1.5 (即亏损150%，投入的这部分财富将损失1.5倍)
            #  - 60%的概率下，有风险资产收益率为  1.5 (即盈利150%，投入的这部分财富将增值1.5倍)
            # 收益率为 -1.5 意味着若投入100元在风险资产，该周期末将变为 -50 元 (损失超过全部本金)。
            # 换言之，若初始财富为 10，该情形下财富将变为 -5 元
            # 收益率为 1.5 则表示若投入100元在风险资产，该周期末将变为 250 元 (翻倍且有额外150%的收益)。
            # 相比之下，若初始财富为 10，这种收益率下财富将变为 25 元
        if action_space is None:
            action_space = [0, 1]  # 默认的可选择投资比例: 0 表示全额投入无风险资产，1 表示全额投入有风险资产
            # 这里简化为只有两种极端选择; 实际中可扩展 action_space 以包含更多比例 (例如 0.5 表示一半资金投入风险资产)

        # 将参数保存为实例属性，供后续使用
        self.aversion_rate = aversion_rate        # 风险厌恶系数
        self.riskless_return = riskless_return    # 无风险资产的收益率 (每期固定增长率)
        self.risky_return = risky_return          # 有风险资产的收益率分布 (概率 -> 收益率)
        self.action_space = action_space          # 动作空间 (投资在有风险资产的比例选项列表)
        self.initial_wealth = initial_wealth      # 初始财富值
        self.T = T                                # 总时间步数 (投资期数)

        # 初始化状态空间和状态转移概率结构
        # state_space: 存储各时间步可能的财富状态 (键为时间 t, 值为该时间可能的财富值集合)
        # 初始时刻 t=0 的财富状态只有 initial_wealth
        # 注意: 时间 t 也是状态维度之一。不同时间步可能对应不同决策策略，因此需要将时间考虑进状态空间
        self.state_space = {t: set() for t in range(T+1)}
        self.state_space[0].add(initial_wealth)
        # transition_prob: 记录状态转移的概率和结果，通过 _generate_states_and_transitions() 填充
        self.transition_prob = self._generate_states_and_transitions()

    def _generate_states_and_transitions(self):
        # 生成所有可能的状态转移及其概率和奖励信息
        # 返回一个嵌套字典 transition_dict:
        # transition_dict[t][wealth][action] = [(prob, next_wealth, reward, done), ...]
        # 表示在时间 t 状态 wealth 下执行 action 后的可能结果:
        #    prob: 该结果发生的概率 (来源于 risky_return 字典)
        #    next_wealth: 下一时间步的财富值 (根据当前财富、所选动作和随机收益计算得到)
        #    reward: 奖励值 (如果此转移后 episode 结束，则为最终财富的效用值; 否则为 0)
        #    done: 是否在此转移后达到终止状态 (True 表示此动作后已是最后一期)
        transition_dict = {}
        for t in range(self.T):
            transition_dict[t] = {}  # 初始化时间 t 的转移信息字典
            for wealth in self.state_space[t]:
                transition_dict[t][wealth] = {}  # 初始化当前财富状态下各动作的转移列表
                for action in self.action_space:
                    transitions = []  # 收集当前状态执行该动作的所有可能转移
                    # 根据有风险资产的收益率分布，生成下一状态的所有可能情况
                    for prob, risky_r in self.risky_return.items():
                        # 计算下一期的财富值 next_wealth
                        # 公式: next_wealth = wealth * [action * (1 + risky_r) + (1 - action) * (1 + riskless_return)]
                        # 解释:
                        #   当前财富 wealth 分为两部分:
                        #   action 部分 (比例) 投入有风险资产，剩余 (1 - action) 部分投入无风险资产
                        #   有风险部分在本期末的值: wealth * action * (1 + risky_r)
                        #   无风险部分在本期末的值: wealth * (1 - action) * (1 + self.riskless_return)
                        #   将两部分相加得到 next_wealth (本期末的总财富)
                        next_wealth = np.round(
                            wealth * (action * (1 + risky_r) + (1 - action) * (1 + self.riskless_return)),
                            3
                        )
                        # 使用 np.round(..., 3) 将结果保留3位小数，避免浮点误差并限制状态空间规模
                        self.state_space[t+1].add(next_wealth)  # 将 next_wealth 加入下一时间步的状态空间
                        # 判断是否达到终止状态:
                        # 如果当前时间步 t 是 T-1 (最后一个决策期)，则执行完动作后 episode 结束
                        done = (t == self.T - 1)
                        # 确定奖励:
                        # 若 done=True (最后一期结束)，奖励为 next_wealth 对应的效用值 (通过 CARA 函数)
                        # 若 done=False (非最后一期)，奖励为 0 (中间过程不计奖励，最终统一计算)
                        reward = self.cara_reward(next_wealth) if done else 0
                        transitions.append((prob, next_wealth, reward, done))
                    # 将当前 wealth 状态下执行 action 的转移列表存入字典
                    transition_dict[t][wealth][action] = transitions
        # 循环结束后，transition_dict 即包含了整个环境的状态转移模型
        # 这定义了在任意状态 (时间 t, 财富值) 下采取任意动作时环境如何转移 (形成一个有限时域的 MDP 模型)
        return transition_dict

    def reset(self):
        # 重置环境状态:
        self.current_wealth = self.initial_wealth  # 当前财富重置为初始财富值
        self.time_step = 0                         # 时间步计数重置为 0 (开始时刻)
        return self.current_wealth                 # 返回初始财富状态

    def step(self, action):
        # 根据当前状态 (时间 self.time_step 下的 self.current_wealth) 和动作 action，模拟环境向前推进一步
        transitions = self.transition_prob[self.time_step][self.current_wealth][action]
        # transitions 是一个列表，包含若干元组 (prob, next_wealth, reward, done)
        # 依据这些概率随机选择一个结果，模拟现实中的不确定性
        probs = [t[0] for t in transitions]                      # 提取每个可能结果的概率
        idx = np.random.choice(len(transitions), p=probs)        # 根据概率分布随机选择一个结果索引
        _, next_wealth, reward, done = transitions[idx]          # 获取选定的结果 (忽略第一个概率值，因为已用其决定选择)
        # 更新当前财富和时间步为转移后的状态
        self.current_wealth = next_wealth
        self.time_step += 1
        # 返回 (下一财富状态, 奖励, 是否终止, 额外信息)
        # 其中额外信息字典在此环境中未使用，留空即可
        return next_wealth, reward, done, {}

    def cara_reward(self, wealth):
        # 计算恒定绝对风险厌恶 (CARA) 型效用函数值作为奖励
        # 函数形式: U(wealth) = - (exp(- aversion_rate * wealth)) / aversion_rate
        # 特点:
        #   - 单调递增: 财富越高，U(wealth) 越大 (注意 U 值为负，但其绝对值越小代表效用越高，即越接近0越好)
        #   - 风险厌恶性: 一阶导数为 exp(-aversion_rate * wealth) > 0，二阶导数为 -aversion_rate * exp(-aversion_rate * wealth) < 0
        #                 边际效用递减，表明随着财富增加，每增加一单位财富带来的效用增益在下降 (体现风险厌恶倾向)
        #   - aversion_rate 越大，效用曲线越陡峭，投资者越厌恶风险 (财富波动对效用影响更大)
        # 在最终一步 (episode 结束时) 使用该效用值作为奖励，从而鼓励代理人最大化最终财富的效用 (而非财富的绝对值)
        return (-np.exp(- self.aversion_rate * wealth)) / self.aversion_rate

# 定义 SARSA 算法的智能体类
# 该智能体通过与 AssetEnv 环境交互，学习状态-动作价值 (Q值) 并改进策略 (policy)，以期逐步逼近最优策略
# SARSA (State-Action-Reward-State-Action) 算法是一种 on-policy 的强化学习方法:
# (名称来源于更新中用到的序列: 状态-动作-奖励-下一个状态-下一个动作)
# 智能体在每个时间步按照当前策略选择动作 (这一步可能是探索或利用)，
# 然后依据执行的动作及其结果更新当前状态-动作的价值估计，策略也随之改进。
class SARSAAgent:
    def __init__(self, env, discount_factor=1, epsilon=0.8, alpha=0.7):
        # env: 环境实例，与智能体交互的资产投资环境 (AssetEnv)
        # discount_factor (γ): 折扣因子，0 <= γ <= 1，决定未来奖励在当前价值评估中的权重
        # 默认值 1 表示不忽略任何未来奖励 (无折扣，完全考虑长期回报)
        # epsilon (ε): 探索率，0 <= ε <= 1，决定采取非最优动作 (探索) 的概率
        # 默认值 0.8 表示有 80% 的时间随机探索动作，20% 的时间选择当前最优动作 (利用)
        # alpha (α): 学习率，0 <= α <= 1，控制每次价值更新的快慢
        # 默认值 0.7 表示每次更新用 70% 的新信息纠正 Q 值 (较高的 α 使学习更快，但可能导致较大波动)
        self.env = env
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.alpha = alpha

        # 初始化 Q 表 (Q-values) 和 策略 (policy)
        self.Q = {}
        self.policy = {}
        # 状态定义为 (t, wealth) 对，其中 t 为时间步，wealth 为该时间步的财富值
        # 仅考虑 t=0 到 T-1 的状态 (t=T 时已终止，无需决策)，为每个状态的每个动作初始化 Q=0，策略为均匀分布
        for t in range(env.T):
            self.Q[t] = {}
            self.policy[t] = {}
            for wealth in env.state_space[t]:
                self.Q[t][wealth] = {a: 0.0 for a in env.action_space}    # 初始化 Q 值为 0
                self.policy[t][wealth] = self._init_policy(env.action_space)  # 初始化策略为均匀随机分布

    def _init_policy(self, actions):
        """ 初始化均匀随机策略 """
        # 返回一个字典，将提供的每个动作的选择概率均等分配 (每个动作概率 = 1 / 动作数)
        return {a: 1/len(actions) for a in actions}

    def _update_policy(self, t, wealth):
        """ 使用 ε-贪婪 (epsilon-greedy) 策略更新给定状态下的策略 """
        # 基于当前 Q 值选取最优动作
        best_action = max(self.Q[t][wealth], key=self.Q[t][wealth].get)
        # 按 ε-贪婪策略重新分配各动作概率:
        #   最优动作的概率 = 1 - ε + ε/|A| (主要选择最优动作，但留出 ε 部分用于探索)
        #   其他动作的概率 = ε/|A| (将 ε 在所有动作上均匀分配，用于探索非最优动作)
        for a in self.policy[t][wealth]:
            if a == best_action:
                self.policy[t][wealth][a] = 1 - self.epsilon + self.epsilon/len(self.env.action_space)
            else:
                self.policy[t][wealth][a] = self.epsilon/len(self.env.action_space)
        # 注: 当 ε=0 时策略为纯贪心 (总是选择 Q 值最大的动作); 当 ε=1 时策略变为完全随机

    def choose_action(self, t, wealth):
        # 根据状态 (t, wealth) 下的策略概率分布随机选择一个动作
        # np.random.choice 会依据给定的概率分布从动作列表中选择一个动作
        selected_action = np.random.choice(
            list(self.policy[t][wealth].keys()),
            p=list(self.policy[t][wealth].values())
        )
        # 该选择包括 ε 的概率随机探索和 (1-ε) 的概率选择当前最优行动
        return selected_action

    def train(self, episodes=10000):
        # 训练智能体：让智能体与环境交互指定次数的独立序列 (episodes)
        # 每个 episode 从环境初始状态开始，不断决策和执行动作，直至达到终止状态，然后进行下一个 episode
        for _ in range(episodes):
            # 重置环境到初始状态，获取初始财富
            wealth = self.env.reset()
            t = 0
            # 根据当前策略在初始状态选择一个动作
            action = self.choose_action(t, wealth)
            done = False

            # 模拟一个 episode 中各时间步的决策过程
            while not done:
                # 执行当前动作，与环境交互得到下一财富状态、即时奖励和是否结束标志
                next_wealth, reward, done, _ = self.env.step(action)
                
                if done:
                    # 如果此动作后 episode 结束 (达到最后一期):
                    # 时间差分目标 TD_target = 当期收到的最终奖励 (因为没有下一状态)
                    td_target = reward
                    # 计算 TD 误差 并用学习率 α 更新 Q 值:
                    # Q[t][wealth][action] <- Q[t][wealth][action] + α * (TD_target - Q[t][wealth][action])
                    self.Q[t][wealth][action] += self.alpha * (td_target - self.Q[t][wealth][action])
                    # 更新策略: 在状态 (t, wealth) 下根据新的 Q 值分布调整策略 (ε-贪婪)
                    self._update_policy(t, wealth)
                    # 结束当前 episode
                    break
                
                # 若未结束，基于策略选择下一状态下的动作 (体现 SARSA 的 on-policy 特性)
                next_action = self.choose_action(t+1, next_wealth)
                # 注意: 这里使用 SARSA 算法，利用策略实际选择的 next_action 的 Q 值来更新 (on-policy)
                # 而 Q-learning 则采用下一状态价值最高的动作来更新 (off-policy)
                # 计算 TD 目标: 当前奖励 + γ * Q[t+1][next_wealth][next_action]
                td_target = reward + self.gamma * self.Q[t+1][next_wealth][next_action]
                # 计算 TD 误差并更新当前 Q 值向目标逼近
                self.Q[t][wealth][action] += self.alpha * (td_target - self.Q[t][wealth][action])
                # 调整策略: 由于 Q 值改变，在状态 (t, wealth) 下按 ε-贪婪原则更新动作概率
                self._update_policy(t, wealth)
                # 状态转移: 进入下一时间步
                wealth, action, t = next_wealth, next_action, t + 1

        # 所有训练回合完成后，返回学习到的 Q 表和策略
        # policy 字典包含每个状态下各动作的最终选择概率 (即智能体学到的策略)
        # Q 表记录每个状态-动作对的价值估计，可用于评估策略或决策参考
        return self.Q, self.policy
    

env = AssetEnv(initial_wealth=10, T=10, aversion_rate=0.01, riskless_return=1,
                  risky_return = {0.4: 0.01, 0.6: 0.02}, action_space=None)
agent = SARSAAgent(env)
Q, policy = agent.train(episodes=1000000)
# 输出最终策略示例
hist = []
for t in range(env.T):
    print(f"Time {t}:")
    for wealth in sorted(env.state_space[t]):
        best_action = max(policy[t][wealth], key=policy[t][wealth].get)
        hist.append(best_action)
        print(f"  Wealth {wealth:.2f}: Best Action {best_action}")

# 计算每个时间步动作的平均值
action_distribution = {}
for t in range(env.T):
    actions_at_t = []
    for wealth in env.state_space[t]:
        best_action = max(policy[t][wealth], key=policy[t][wealth].get)
        actions = action_distribution.get(t, [])
        actions_at_t = actions + [best_action]
        action_distribution[t] = actions_at_t

# 计算每个时间步动作的平均
avg_actions = [np.mean(action_distribution[t]) for t in range(env.T)]

# 绘制平均动作随时间变化的图表
plt.figure(figsize=(12, 6))
plt.plot(range(env.T), avg_actions, marker='o', linestyle='-', color='b')

plt.xlabel('Time')
plt.ylabel('Average Action (Risky Asset Allocation)')
plt.title('Average Optimal Action over Time (SARSA Policy)')
plt.grid(True)
plt.show()
