# Continuous Control with deep reinforcement learning
https://arxiv.org/abs/1509.02971
## Introduction
Model-free off-policy actor-critic algorithm using deep function approximators that can learn policies in high-dimensional, contiuous action spaces. </br>

## DQN
DQN is able to learn value functions using usch function approximators in a stable and robust way due to two innovations.
- The network is trained off-policy with samples from a replay buffer to minimize correlations between samples.
- The network is trained with a Q network to give consistent targets during temporal difference backups.
- In order to scale Q-learning, we have two changes:
1. replay buffer
	- A finite size cache, (S, A, R, St+1).
	- At each timestep, the actor and critic are updated by sampling a minibatch uniformly from the buffer.

2. separate target network

## Impoosible to directly apply Q-learning to continuous action spaces
- In continuous spaces finding the greedy policy requires an optimization of a at every timestep.
- This Optimization is too slow to be practical with large, unconstrained function approximators and nontrivial action spaces.

## DPG
DPG maintains a parameterized actor function.. which specifies the current policy by deterministically mapping states to a specific action.
- The Critic Q(s, a) is learned using bellman equation as in Q-learning
- The Actor is updated by following the applying the chain rule to expected return from the start distribution J with respect to the actor parameters

## DDPG
### Challenge when using NN for RL
- Most optimization algorithms assume that the samples are independently and identically distributed.
- To make efficient use of hardware optimizations, It is essential to learn in **minibatch**, rather than online
- We create a copy of the actor and critic networks, Q and u respectively, that are used to calculating the target values. The weights of these target networks are then updated by having them slowly track the learned networks. This means that thae target values are constrained to change slowly, greatly improving the stability of learning.
Deep DPG can learn competitive policies for all of our tasks using low-dimensional observations using the same hyper-parameters and network structure.
- It requires only straightforward actor-critic architecture and learning algorithm with very few "moving parts"

## Implementation
### Hyperparameters
```python
# -------------------
# Training Parameters　For Pendulum-v0
# -------------------
MAX_EPISODES = 1000
MAX_EP_STEPS = 1000
NOISE_MAX_EP = 200
# Noise parameters - Ornstein Uhlenbeck
DELTA = 0.5
SIGMA = 0.5
OU_A = 3.
OU_MU = 0.
# Reward parameters
REWARD_FACTOR = 0.01 # Discrete: Reward factor = 0.1
# Base learning rate for the actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic network
CRITIC_LEARNING_RATE = 0.001
# Discount Factor
GAMMA = 0.99
# soft target update param
TAU = 0.001

# -------------------
# Unility Parameters
# -------------------
RANDOM_SEED = 1234
BUFFER_SIZE = 100000
MINIBATCH_SIZE = 100
```
### ActorNetwork
```python
inputs = tf.placeholder(tf.float32, [None, self.s_dim])
        
# Input -> Hidden Layer
w1 = weight_variable([self.s_dim, n_hidden_1])
b1 = bias_variable([n_hidden_1])
# Hidden Layer -> Hidden Layer
w2 = weight_variable([n_hidden_1, n_hidden_2])
b2 = bias_variable([n_hidden_2])

# Hidden Layer -> Output
w3 = weight_variable([n_hidden_2, self.a_dim])
b3 = bias_variable([self.a_dim])

# 1st hidden layer, option: Softmax, ReLU, tanh or sigmoid
h1 = tf.nn.relu(tf.matmul(inputs, w1) + b1)
 
# 2nd hidden layer, option: Softmax, ReLU, tanh or sigmoid
h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

# Run tanh on output to get -1 to 1
out = tf.nn.tanh(tf.matmul(h2, w3) + b3)
# Scale output to -action_bound to action_bound
scaled_out = tf.multiply(out, self.action_bound)
```
### CriticNetwork
```python
inputs = tf.placeholder(tf.float32, [None, self.s_dim])
action = tf.placeholder(tf.float32, [None, self.a_dim])

# Input -> Hidden Layer
w1 = weight_variable([self.s_dim, n_hidden_1])
b1 = bias_variable([n_hidden_1])
# Hidden Layer -> Hidden Layer + Action
w2 = weight_variable([n_hidden_1, n_hidden_2])
w2a = weight_variable([self.a_dim, n_hidden_2])
b2 = bias_variable([n_hidden_2])

# Hidden Layer -> Output Q-value
w3 = weight_variable([n_hidden_2, 1])
b3 = bias_variable([1])

# 1st hidden layer, option: Softmax, ReLU, tanh or sigmoid
h1 = tf.nn.relu(tf.matmul(inputs, w1) + b1)
 
# 2nd hidden layer, option: Softmax, ReLU, tanh or sigmoid
# Action inserted here.
h2 = tf.nn.relu(tf.matmul(h1, w2) + tf.matmul(action, w2a) + b2)

out = tf.matmul(h2, w3) + b3
```
### Test envs
CartPole-v0