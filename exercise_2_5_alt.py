import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.DEBUG, filename='exercise_2_5.log')
    
def random_walk(q_set, mean=0, std=1):
    dim = np.shape(q_set)
    return q_set + np.random.normal(np.zeros(dim)+mean, std)

def next_action(action_set, eps=0.1):
    n_actions, n_tasks = np.shape(action_set)
    explore = np.random.uniform(size=n_tasks) < eps
    actions = np.argmax(action_set, axis = 0)
    if any(explore):
        exp_actions = np.random.randint(n_actions, size=n_tasks)
        actions = np.where(explore, exp_actions, actions)
    return actions

def calculate(q, Q, action_id_map):
    actions = next_action(Q)
    rewards = np.choose(actions, q)
    action_mask = np.equal(action_id_map, actions)
    np.add(np.array(np.zeros(np.shape(action_id_map))), rewards, out=rewards_matrix, where=action_mask)
    return rewards, rewards_matrix, action_mask


epsilon = 0.1
alpha_cs = 0.1
n_iterations = 1000
n_actions = 10
n_tasks = 500
action_id_map = np.transpose([list(range(n_actions))] * n_tasks)
Q_sa = np.zeros((n_actions, n_tasks)) 
Q_cs = np.zeros((n_actions, n_tasks))
q = np.zeros((n_actions, n_tasks))
q_sa_n = np.zeros((n_actions, n_tasks))
q_cs_n = np.zeros((n_actions, n_tasks))
reward_sa_avg = np.zeros(n_iterations)
reward_cs_avg = np.zeros(n_iterations)

beta = np.zeros((n_actions, n_tasks))
rewards_matrix = np.zeros((n_actions, n_tasks))

for i in range(n_iterations):
    q = random_walk(q)
    
    rewards, rewards_matrix, action_mask = calculate(q, Q_cs, action_id_map)
    reward_cs_avg[i] = np.average(rewards)
    np.add(q_cs_n, 1, out=q_cs_n, where=action_mask)
    Q_cs = Q_cs + alpha_cs * (rewards_matrix - Q_cs)
    
    rewards, rewards_matrix, action_mask = calculate(q, Q_sa, action_id_map)
    reward_sa_avg[i] = np.average(rewards)
    np.add(q_sa_n, 1.0, out=q_sa_n, where=action_mask)
    
    np.divide(1.0, q_sa_n, out=beta, where=action_mask)
    Q_sa = Q_sa + beta * (rewards_matrix - Q_sa)

logging.info("Q_sa: \n%s", np.array2string(Q_sa, precision=4, floatmode='fixed'))
logging.info("Q_cs: \n%s", np.array2string(Q_cs, precision=4, floatmode='fixed'))
logging.info("q: \n%s", np.array2string(q, precision=4, floatmode='fixed'))

plt.plot(reward_sa_avg, label='Sample Average', color='r')
plt.plot(reward_cs_avg, label='Constant Step Size', color='b')
plt.legend()
plt.show()
