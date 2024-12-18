import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.DEBUG, filename='exercise_2_5.log')

class TenArms:
    def __init__(self, n_actions=10, n_tasks=500, random_walk_mean=0, random_walk_std=1):
        self.arms = np.zeros((n_actions, n_tasks))
        self.random_walk_mean = random_walk_mean
        self.random_walk_std = random_walk_std
        self.zeros = np.zeros((n_actions, n_tasks))
        
    def random_walk_arms(self):
        self.arms = self.arms + np.random.normal(self.zeros+self.random_walk_mean, self.random_walk_std)

class TenArmedBandit:
    def __init__(self, n_actions=10, n_tasks=500, epsilon=0.1, constant_step_size=None):
        self.n_actions = n_actions
        self.n_tasks = n_tasks
        self.shape = (n_actions, n_tasks)
        self.epsilon = epsilon
        self.constant_step_size = constant_step_size
        self.action_id_map = np.transpose([list(range(n_actions))] * n_tasks)
        self.Q = np.zeros((n_actions, n_tasks))
        self.q_n = np.zeros((n_actions, n_tasks))
        self.rewards_matrix = np.zeros((n_actions, n_tasks))

    def iterate(self, arms):
        actions = self.next_action()
        rewards = np.choose(actions, arms)
        action_mask = np.equal(self.action_id_map, actions)
        np.add(np.zeros(self.shape), rewards, out=self.rewards_matrix, where=action_mask)
        np.add(self.q_n, 1.0, out=self.q_n, where=action_mask)
        if self.constant_step_size:
            self.Q = self.Q + self.constant_step_size * (self.rewards_matrix - self.Q)
        else:
            beta = np.zeros(self.shape)
            np.divide(1.0, self.q_n, out=beta, where=action_mask)
            self.Q = self.Q + beta * (self.rewards_matrix - self.Q)
        reward_avg = np.average(rewards)
        return reward_avg

    def next_action(self):
        explore = np.random.uniform(size=self.n_tasks) < self.epsilon
        actions = np.argmax(self.Q, axis = 0)
        if any(explore):
            exp_actions = np.random.randint(self.n_actions, size=self.n_tasks)
            actions = np.where(explore, exp_actions, actions)
        return actions

def main(n_iterations=1000):
    ten_arms = TenArms()
    ten_armed_bandit_sample_avg = TenArmedBandit()
    ten_armed_bandit_constant_step_size = TenArmedBandit(constant_step_size=0.1)
    
    reward_sa_avg = np.zeros(n_iterations)
    reward_cs_avg = np.zeros(n_iterations)

    for i in range(n_iterations):
        ten_arms.random_walk_arms()
        reward_sa_avg[i] = ten_armed_bandit_sample_avg.iterate(ten_arms.arms)
        reward_cs_avg[i] = ten_armed_bandit_constant_step_size.iterate(ten_arms.arms)
        
    logging.info("Q_sa: \n%s", np.array2string(ten_armed_bandit_sample_avg.Q, precision=4, floatmode='fixed'))
    logging.info("Q_cs: \n%s", np.array2string(ten_armed_bandit_constant_step_size.Q, precision=4, floatmode='fixed'))

    plt.plot(reward_sa_avg, label='Sample Average', color='r')
    plt.plot(reward_cs_avg, label='Constant Step Size', color='b')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()
