# Deep Q-network implementation in cartpole of openAI gym
# 2017.07.02
# Xiao Tang

import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque
from gym import wrappers

# Hyper parameters for DQN
GAMMA = 0.9 # discount facter
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # minibatch size

# DQN class
class DQN():
    """
    Deep Q-network class
    """
    def __init__(self, env):
        # experience replay
        self.replay_buffer = deque()
        # init parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.create_Q_network()
        self.create_training_method()
        
        # init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def create_Q_network(self):
        """
        create a Q-learning network
        """
        # network weights 
        W1 = self.weight_variable([self.state_dim, 20])
        b1 = self.bias_variable([20])
        W2 = self.weight_variable([20, self.action_dim])
        b2 = self.bias_variable([self.action_dim])
        
        # input layer
        # we use the format [None, state_dim] cuz minibatch
        self.state_input = tf.placeholder("float", [None, self.state_dim])
        # hidden layers
        h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
        # Q value layer
        self.Q_value = tf.matmul(h_layer, W2) + b2
    
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def create_training_method(self):
        """
        
        """
        self.action_input = tf.placeholder("float", [None, self.action_dim]) # one hot
        self.y_input = tf.placeholder("float", [None]) # target Q value
        # reduce sum: compress data to 1-d
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)
    
    def perceive(self, state, action, reward, next_state, done):
        """
        Perceive to store information,
        We start training when storage data > batch size
        In MLP, we use one_hot_key, however in openAI gym, we use single value
        For example, output action 1 -> [0, 1], 0 -> [1, 0]
        """
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()
        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()
        
    def train_Q_network(self):
        self.time_step += 1

        # step 1. obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
    
        # step 2. calculate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict = {self.state_input: next_state_batch})
        for i in range(BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch       
        })        

    def egreedy_action(self, state):
        Q_value = self.Q_value.eval(feed_dict = {self.state_input:[state]})[0]
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(Q_value)
        
        self.epilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000


    def action(self, state):  
        return np.argmax(self.Q_value.eval(feed_dict = {self.state_input:[state]})[0])


# -----------------------------------------
# Hyper Parameters
ENV_NAME = "CartPole-v0"
EPISODE = 3000 # episode limitation
STEP = 3000 # step limitation in an episode
TEST = 10
result_file = "cartpole-experiment-3"
UPLOAD = True

def main():
    # init openAI gym env and dqn agent
    env = gym.make(ENV_NAME)
    agent = DQN(env)
    
    env = wrappers.Monitor(env, result_file, force=True)
    for episode in xrange(EPISODE):
        
        # init task
        state = env.reset()
        # Training
        total_reward = 0.0
        for step in xrange(STEP):
            action = agent.egreedy_action(state) # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            # Define reward for agent
            reward_agent = -1 if done else 0.1
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        # test every 100 episode
        if episode % 100 == 0: 
            print "episode: %d, Evaluation Average Reward: %f" % (episode, total_reward)

    if UPLOAD:
        gym.upload(result_file, api_key="sk_JObiOSHpRjw48FpWvI1GA")
    
    """
    for i in xrange(100):
        state = env.reset()
        for j in xrange(200):
            env.render()
            action = agent.action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
    """
if __name__ == "__main__":
    main()
    

